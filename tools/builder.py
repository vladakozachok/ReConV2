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
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) #Shuffle
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs,
                                                 num_workers=int(args.num_workers),
                                                 drop_last=config.others.subset == 'train',
                                                 worker_init_fn=worker_init_fn,
                                                 pin_memory=True,
                                                 sampler=sampler)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs,
                                                 shuffle=False,
                                                 drop_last=config.others.subset == 'train',
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

# import torch
# import random
# from collections import defaultdict
# from torch.utils.data import DataLoader, Sampler

# # Custom batch sampler that groups samples using the 2222 / 233 logic.
# class CustomBatchSampler(Sampler):
#     def __init__(self, dataset):
#         # Build mapping from asset_id to list of sample indices.
#         self.asset2indices = defaultdict(list)
#         for idx, sample in enumerate(dataset):
#             asset_id = sample["asset_id"]  # each sample must have an "asset_id" field
#             self.asset2indices[asset_id].append(idx)
#         # Only consider assets that have at least 11 augmentations;
#         # if an asset has more than 11, we only keep the first 11 (after shuffling).
#         self.assets = []
#         for asset, indices in self.asset2indices.items():
#             if len(indices) >= 11:
#                 random.shuffle(indices)  # shuffle indices within the asset
#                 self.asset2indices[asset] = indices[:11]
#                 self.assets.append(asset)
#         # Shuffle the list of eligible assets.
#         random.shuffle(self.assets)

#     def __iter__(self):
#         # Make a working copy of the assets list.
#         assets_list = self.assets.copy()
#         batches = []
#         while len(assets_list) >= 4:
#             # --- Batch Type 1: 2-2-2-2 ---
#             # Select 4 unique assets at random.
#             batch1_assets = random.sample(assets_list, 4)
#             batch1 = []
#             for asset in batch1_assets:
#                 # Each asset contributes its first 2 samples.
#                 batch1.extend(self.asset2indices[asset][:2])
#                 self.asset2indices[asset] = self.asset2indices[asset][2:]
#                 # If the asset no longer has at least 2 samples for a 2222 batch, remove it.
#                 if len(self.asset2indices[asset]) < 2:
#                     assets_list.remove(asset)
#             batches.append(batch1)

#             # --- Batch Type 2: 2-3-3 ---
#             if len(assets_list) >= 3:
#                 batch2_assets = random.sample(assets_list, 3)
#                 batch2 = []
#                 # First asset contributes 2 samples.
#                 batch2.extend(self.asset2indices[batch2_assets[0]][:2])
#                 self.asset2indices[batch2_assets[0]] = self.asset2indices[batch2_assets[0]][2:]
#                 # Next two assets contribute 3 samples each.
#                 for asset in batch2_assets[1:]:
#                     batch2.extend(self.asset2indices[asset][:3])
#                     self.asset2indices[asset] = self.asset2indices[asset][3:]
#                     if len(self.asset2indices[asset]) < 3:
#                         assets_list.remove(asset)
#                 batches.append(batch2)
#         # Finally, shuffle the order of the batches.
#         random.shuffle(batches)
#         for batch in batches:
#             yield batch

#     def __len__(self):
#         # It can be challenging to compute the exact number of batches beforehand.
#         return 0  # This value is not used by DataLoader when using batch_sampler.

# # Integration into dataset_builder.
# def dataset_builder(args, config):
#     dataset = build_dataset_from_cfg(config._base_, config.others)
#     shuffle = config.others.subset == 'train'

#     if config.get('extra_train') is not None:
#         config.extra_train.others.img = config.others.img
#         config.extra_train.others.text = config.others.text
#         config.extra_train.others.img_views = config.others.img_views
#         extra_dataset = build_dataset_from_cfg(config.extra_train._base_, config.extra_train.others)
#         dataset += extra_dataset

#     # Create our custom batch sampler from the dataset.
#     custom_batch_sampler = CustomBatchSampler(dataset)

#     if args.distributed:
#         # When distributed, you might still need a DistributedSampler.
#         # One option is to wrap your dataset with DistributedSampler first
#         # and then apply your custom batching on top of that.
#         sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
#         dataloader = DataLoader(dataset, batch_sampler=custom_batch_sampler,
#                                 num_workers=int(args.num_workers),
#                                 drop_last=config.others.subset == 'train',
#                                 worker_init_fn=worker_init_fn,
#                                 pin_memory=True)
#     else:
#         sampler = None
#         dataloader = DataLoader(dataset, batch_sampler=custom_batch_sampler,
#                                 num_workers=int(args.num_workers),
#                                 drop_last=config.others.subset == 'train',
#                                 pin_memory=True,
#                                 worker_init_fn=worker_init_fn)
#     return sampler, dataloader