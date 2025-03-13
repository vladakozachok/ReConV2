# optimizer
import torch.optim as optim
# dataloader
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
# utils
from utils.logger import *
from utils.misc import *
from timm.scheduler import CosineLRScheduler

def dataset_builder(args, config):
    dataset = build_dataset_from_cfg(config._base_, config.others)
    shuffle = config.others.subset == 'train'

    if config.get('extra_train') is not None:
        config.extra_train.others.img = config.others.img
        config.extra_train.others.text = config.others.text
        config.extra_train.others.img_views = config.others.img_views
        extra_dataset = build_dataset_from_cfg(config.extra_train._base_, config.extra_train.others)
        dataset += extra_dataset

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        dataloader = DataLoader(dataset,
                                batch_size=config.others.bs,
                                num_workers=int(args.num_workers),
                                drop_last=config.others.subset == 'train',
                                worker_init_fn=worker_init_fn,
                                pin_memory=True,
                                sampler=sampler)
    else:
        sampler = None
        # Use the custom batch sampler that ensures each batch has a similar pair.
        batch_sampler = SimilarPairBatchSampler(dataset, batch_size=config.others.bs)
        dataloader = DataLoader(dataset,
                                batch_sampler=batch_sampler,
                                num_workers=int(args.num_workers),
                                pin_memory=True,
                                worker_init_fn=worker_init_fn)
    return sampler, dataloader

def model_builder(config):
    model = build_model_from_cfg(config)
    return model


def build_opti_sche(base_model, config):
    opti_config = config.optimizer
    opti_config.kwargs.lr = float(opti_config.kwargs.lr)
    lr = opti_config.kwargs.lr
    weight_decay = opti_config.kwargs.weight_decay
    skip_list = ()

    decay = []
    no_decay = []
    finetune_head = []
    for name, param in base_model.module.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if 'cls' in name and config.model.NAME == 'PointTransformer':
            print("10 * LR: ", name)
            finetune_head.append(param)
        elif len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    param_groups = [
        {'params': no_decay, 'weight_decay': 0., 'lr': lr},
        {'params': decay, 'weight_decay': weight_decay, 'lr': lr},
        {'params': finetune_head, 'lr': lr * 10}
    ]
    momentum_value = opti_config.get('momentum', 0.9)  # Default to 0.9 if not specified
    dampening_value = 0  # Nesterov momentum requires dampening to be 0
    if opti_config.type == 'AdamW':
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs, eps = 1e-4, amsgrad=True)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(param_groups, **opti_config.kwargs, eps = 1e-4, amsgrad=True)
    elif opti_config.type == 'RAdam':
        optimizer = optim.RAdam(param_groups, **opti_config.kwargs, eps=1e-4, amsgrad=True)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(param_groups, momentum=momentum_value, dampening=dampening_value, nesterov=True, **opti_config.kwargs)
    else:
        raise NotImplementedError()

    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs)  # misc.py
    elif sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=sche_config.kwargs.epochs,
                                      lr_min=1e-6,
                                      warmup_lr_init=1e-6,
                                      warmup_t=sche_config.kwargs.initial_epochs,
                                      cycle_limit=1,
                                      t_in_epochs=True)
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config.kwargs)
    elif sche_config.type == 'function':
        scheduler = None
    else:
        raise NotImplementedError()

    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
        scheduler = [scheduler, bnscheduler]

    return optimizer, scheduler


def resume_model(base_model, args, logger=None):

    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger=logger)
        return 0, 0
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger=logger)

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    # parameter resume of base model
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}

    base_model.load_state_dict(base_ckpt, strict=True)

    # parameter
    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()

    print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})',
              logger=logger)
    return start_epoch, best_metrics


def resume_optimizer(optimizer, args, logger=None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger=logger)
        return 0, 0, 0
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', logger=logger)
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])


def save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, logger=None):
    if args.local_rank == 0:
        model = base_model.module.state_dict() if args.distributed else base_model.state_dict()
        torch.save({
            'base_model': model,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics.state_dict() if metrics is not None else dict(),
            'best_metrics': best_metrics.state_dict() if best_metrics is not None else dict(),
        }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger=logger)


def save_pretrain_model(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, logger=None):
    if args.local_rank == 0:
        model = base_model.module.state_dict() if args.distributed else base_model.state_dict()
        torch.save({
            'base_model': model,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics.state_dict() if metrics is not None else dict(),
            'best_metrics': best_metrics.state_dict() if best_metrics is not None else dict(),
        }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger=logger)


def load_model(base_model, ckpt_path, logger=None):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger=logger)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    base_model.load_state_dict(base_ckpt, strict=True)

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger=logger)
    return

import random
import torch
from torch.utils.data import DataLoader, Sampler

class SimilarPairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=10):
        """
        Args:
            dataset: Your dataset instance. It must have an attribute `datapath` which is a list of tuples,
                     where the first element is the asset identifier.
            batch_size: Number of samples per batch (e.g. 10).
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = len(dataset)
        
        # Build a mapping from asset id to list of indices.
        self.asset_to_indices = {}
        for idx, (asset, _) in enumerate(dataset.datapath):
            self.asset_to_indices.setdefault(asset, []).append(idx)

    def __iter__(self):
        # Copy of all indices that haven't been used yet.
        remaining = set(range(self.num_samples))
        batches = []
        
        while len(remaining) >= self.batch_size:
            # Build candidate list: assets with at least 2 available samples.
            candidate_assets = []
            for asset, indices in self.asset_to_indices.items():
                available = [i for i in indices if i in remaining]
                if len(available) >= 2:
                    candidate_assets.append((asset, available))
            
            pairs = []
            # We need to form two similar pairs (i.e. 4 samples).
            if len(candidate_assets) >= 2:
                # Choose two distinct assets.
                chosen = random.sample(candidate_assets, 2)
                for asset, available in chosen:
                    pair = random.sample(available, 2)
                    pairs.append(pair)
            elif len(candidate_assets) == 1:
                # Only one asset qualifies; see if it has at least 4 available samples.
                asset, available = candidate_assets[0]
                if len(available) >= 4:
                    sampled = random.sample(available, 4)
                    pairs.append(sampled[:2])
                    pairs.append(sampled[2:4])
                else:
                    # Cannot form two pairs from a single asset.
                    break
            else:
                # No asset available to form even one pair.
                break
            
            # Remove the indices chosen for the two pairs from the remaining set.
            for pair in pairs:
                for idx in pair:
                    remaining.remove(idx)
            
            batch = []
            for pair in pairs:
                batch.extend(pair)
            
            # Fill the rest of the batch with random samples from the remaining indices.
            remaining_to_fill = self.batch_size - 4
            if len(remaining) < remaining_to_fill:
                break
            additional = random.sample(remaining, remaining_to_fill)
            for idx in additional:
                remaining.remove(idx)
            batch.extend(additional)
            
            # Shuffle the batch so the pair positions are not fixed.
            random.shuffle(batch)
            batches.append(batch)
        
        # Optionally yield any leftover indices in a final (smaller) batch.
        if remaining:
            batches.append(list(remaining))
        
        for batch in batches:
            yield batch

    def __len__(self):
        return self.num_samples // self.batch_size

