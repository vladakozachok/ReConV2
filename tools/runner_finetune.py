import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms

import wandb

wandb.init(
    project="ReCon2",
)
class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def run_net(args, config):
    logger = get_logger(args.log_name)
    # build dataset
    config.dataset.train.others.with_color = config.model.with_color
    config.dataset.val.others.with_color = config.model.with_color
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
                                                               builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)

    if config.dataset.train._base_.NAME == "ModelNet":
        train_transforms = transforms.Compose([
            data_transforms.PointcloudScaleAndTranslate(),
        ])
    else:
        train_transforms = transforms.Compose([
            data_transforms.PointcloudRotate(),
        ])

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)
    print_log(start_epoch)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger=logger)

    base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    total_params = sum(p.numel() for p in base_model.parameters()) / 1e6
    print_log(f"Total number of parameters: {total_params:.2f}M", logger=logger)
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad) / 1e6
    print_log(f"Total number of trainable parameters: {trainable_params:.2f}M", logger=logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode

        npoints = config.npoints
        
        torch.cuda.empty_cache()

        gradient_norms = []
        save_interval = 100  # Save heatmap every 100 steps
        step_counter = 0                  
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1

            data_time.update(time.time() - batch_start_time)

            points = data[0].cuda()
            label = data[1].cuda()
            name = data[2]

            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points.size(1) < point_all:
                point_all = points.size(1)

            fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
            fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1,
                                                                                        2).contiguous()  # (B, N, 3)
            
            # Normalize the points to range [0, 1]
            min_vals = points.min(dim=1, keepdim=True)[0]  # (B, 1, 3)
            max_vals = points.max(dim=1, keepdim=True)[0]  # (B, 1, 3)

            # Avoid division by zero (if max_vals equals min_vals)
            range_vals = max_vals - min_vals
            range_vals = torch.clamp(range_vals, min=1e-6)  # To prevent division by zero

            # # Normalize to [0, 1]
            # normalized_points = (points - min_vals) / range_vals
            # points = train_transforms(normalized_points)
            
            # print_log("Checking input tensor for NaNs...", logger=logger)
            # print_log(f"NaN in input: {torch.isnan(points).any()}", logger=logger)
            # print_log(f"Max value: {points.max()}, Min value: {points.min()}", logger=logger)
            # print_log(f"Mean value: {points.mean()}, Std: {points.std()}", logger=logger)
            
            ret, cd_loss = base_model(points)
            loss, acc = base_model.module.get_loss_acc(ret, label, name)
            
            if loss is None or acc is None:
                
                print(f"Skipping batch {idx} due to NaN values in embeddings or loss/accuracy.")
                print_log(f"Contracts that give NaN: {name}", logger = logger)
                del points, label, name, ret, cd_loss, loss, acc
                torch.cuda.empty_cache()
                continue 
            
            _loss = loss  + 3 * cd_loss
            _loss.backward()


            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                # clip_value = config.get('grad_norm_clip', 1.0)  # Default clip value = 1.0
                # torch.nn.utils.clip_grad_norm_(base_model.parameters(), clip_value, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

              
            # torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0, norm_type=2)


            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc])
            else:
                losses.update([loss.item(), acc])

            losses_avg = losses.avg()
            
            wandb.log({
                "epoch": epoch,
                "batch_idx": idx,
                "train_loss_avg": losses_avg[0],  # Average loss so far
                "train_accuracy_avg": losses_avg[1],  # Average accuracy so far
                "batch_time": batch_time.avg(),  # Average batch processing time
                "data_time": data_time.avg(),    # Average data loading time
                "learning_rate": optimizer.param_groups[0]["lr"],  # Current learning rate
            })

            if args.distributed:
                torch.cuda.synchronize()

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                   optimizer.param_groups[0]['lr']), logger=logger)

        # if epoch % args.val_freq == 0 and epoch != 0:
        if epoch %1 ==0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, args, config, best_metrics,
                               logger=logger)

            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                        logger=logger)
                print_log(
                    "--------------------------------------------------------------------------------------------",
                    logger=logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
        

            

def validate(base_model, test_dataloader, epoch, args, config, best_metrics, logger=None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    test_name = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()
            name = data[2]

            points = misc.fps(points, npoints)
            embeddings, _ = base_model(points)

            test_pred.append(embeddings)
            test_label.append(label)
            test_name.extend(name)
        

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        test_pred = F.normalize(test_pred, p=2, dim=1)
        sim_matrix = F.cosine_similarity(test_pred.unsqueeze(1), test_pred.unsqueeze(0), dim=-1) 
        sim_matrix.fill_diagonal_(-float('inf'))
        top1_indices = torch.argmax(sim_matrix, dim=1)
        
        labels = test_label.view(-1)  # Flatten labels to shape (N,)
        num_samples = labels.size(0)

        top1_labels = labels[top1_indices]
        correct_top1 = (top1_labels == labels).float().sum().item()
        top1_accuracy = correct_top1 / num_samples * 100.0

        acc = top1_accuracy


        print_log('[Validation Similarity] EPOCH: %d  accuracy = %.4f' % (epoch, acc), logger=logger)

        wandb.log({
                "val_top1_acc": acc,
            })

        if args.distributed:
            torch.cuda.synchronize()
        
        del sim_matrix
    return Acc_Metric(acc)


def validate_vote(base_model, test_dataloader, epoch, args, config, logger=None, times=10):
    print_log(f"[VALIDATION_VOTE] epoch {epoch}", logger=logger)
    base_model.eval()  # set model to eval mode

    total_correct = 0
    total_samples = 0
    npoints = config.npoints
    
    # Set up the transformations based on the dataset
    if config.dataset.train._base_.NAME == "ModelNet":
        test_transforms = transforms.Compose([data_transforms.PointcloudScale()])
    elif config.dataset.train._base_.NAME == "ScanObjectNN_hardest":
        test_transforms = transforms.Compose([data_transforms.PointcloudRotate()])
    else:
        test_transforms = transforms.Compose([])

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()

            # Determine the number of points to sample
            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            # Sample points using furthest point sampling
            points = misc.fps(points_raw, npoints)

            # Collect embeddings across multiple passes
            local_preds = []
            for _ in range(times):
                transformed_points = test_transforms(points)

                # Get embeddings
                embeddings, _ = base_model(transformed_points)
                local_preds.append(embeddings.unsqueeze(0))

            test_pred = F.normalize(test_pred, p=2, dim=1)
            sim_matrix = F.cosine_similarity(test_pred.unsqueeze(1), test_pred.unsqueeze(0), dim=-1)
            sim_matrix.fill_diagonal_(-float('inf'))  # exclude self-similarity

            labels = test_label.view(-1)  # shape: (N,)
            num_samples = labels.size(0)

            # ----- Top-1 Accuracy -----
            top1_indices = torch.argmax(sim_matrix, dim=1)
            top1_labels = labels[top1_indices]
            correct_top1 = (top1_labels == labels).float().sum().item()
            top1_accuracy = correct_top1 / num_samples * 100.0

            # ----- Top-5 Accuracy -----
            top5_indices = torch.topk(sim_matrix, k=5, dim=1).indices  # shape: (N,5)
            # For each sample, check if any of the top-5 predictions has the same label as the anchor
            top5_correct = 0
            for i in range(num_samples):
                if (labels[i] == labels[top5_indices[i]]).any():
                    top5_correct += 1
            top5_accuracy = top5_correct / num_samples * 100.0

            # ----- Mean Positive and Negative Similarities -----
            # Create a boolean mask where True = same label (positive pair) and exclude self
            mask_pos = (labels.unsqueeze(1) == labels.unsqueeze(0))
            mask_self = torch.eye(num_samples, dtype=torch.bool, device=labels.device)
            mask_pos = mask_pos & ~mask_self

            # Mean similarity for positives and negatives
            mean_positive_sim = sim_matrix[mask_pos].mean().item() if mask_pos.sum() > 0 else float('nan')
            mean_negative_sim = sim_matrix[~mask_pos].mean().item() if (~mask_pos).sum() > 0 else float('nan')
            margin = mean_positive_sim - mean_negative_sim

        # Log to wandb
        wandb.log({
            "val_top1_acc": top1_accuracy,
            "val_top5_acc": top5_accuracy,
            "mean_positive_sim": mean_positive_sim,
            "mean_negative_sim": mean_negative_sim,
            "sim_margin": margin
        })

        # Calculate overall accuracy
        if total_samples > 0:
            acc = top1_accuracy
        else:
            acc = 0.0  # Handle edge case where there are no samples

        print_log('[Validation Vote] EPOCH: %d  acc_vote = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    return Acc_Metric(acc)


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger=logger)  # for finetuned transformer
    base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)


def test(base_model, test_dataloader, args, config, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            logits = base_model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('-' * 10, logger=logger)
        print_log('[TEST] acc = %.4f' % acc, logger=logger)
        print_log('-' * 10, logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

        print_log(f"[TEST_VOTE]", logger=logger)
        acc = 0.
        for time in range(1, 300):
            this_acc = test_vote(base_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
            if acc < this_acc:
                acc = this_acc
            print_log('[TEST_VOTE_time %d]  acc = %.4f, best acc = %.4f' % (time, this_acc, acc), logger=logger)
        print_log('[TEST_VOTE] acc = %.4f' % acc, logger=logger)


def test_vote(base_model, test_dataloader, epoch, args, config, logger=None, times=10):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints
    if config.dataset.train._base_.NAME == "ModelNet":
        test_transforms = transforms.Compose([
            data_transforms.PointcloudScale(),
        ])
    elif config.dataset.train._base_.NAME == "ScanObjectNN_hardest":
        test_transforms = transforms.Compose([
            data_transforms.PointcloudRotate(),
        ])
    else:
        test_transforms = transforms.Compose([])
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(),
                                                          fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)

            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.

        if args.distributed:
            torch.cuda.synchronize()

    return acc
