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

import random
from torch.utils.data import Sampler

class SimilarPairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=10, seed=42):
        """
        Args:
            dataset: Your dataset instance. It must have an attribute `datapath` which is a list of tuples,
                     where the first element is the asset identifier (asset path).
            batch_size: Number of samples per batch (should be 10 in this case).
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = len(dataset)
        self.seed = seed
        
        # Initialize mappings for label to indices, contract, and label
        self.label_to_indices = {}
        self.asset_to_contract = {}
        self.asset_to_label = {}
        
        # Assuming dataset.datapath contains the full asset path
        for idx, asset_path in enumerate(dataset.datapath):
            contract, label = self.extract_contract_and_label(asset_path)
            
            # Group samples by label instead of asset_path
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
            
            # Store the contract and label associated with this asset
            self.asset_to_contract[idx] = contract
            self.asset_to_label[idx] = label

    def extract_contract_and_label(self, asset_path):
        """
        Given an asset path, extract the contract and label.
        """
        # Example: Extracting the label from the asset path (customize based on your path structure)
        parts = asset_path[0].split('-')
        
        # The contract is the first part of the asset path
        contract = parts[0]
        
        # The label is the last part before ".pt" (adjust as needed)
        label = asset_path[0]  # assuming label is second-to-last part in the path
        
        return contract, label

    def __iter__(self):
        r = random.Random(self.seed)
        
        # Create a local copy (per epoch) of the available indices per label.
        available = {label: indices.copy() for label, indices in self.label_to_indices.items()}
        
        batches = []
        
        # Continue until we can no longer form a full batch.
        while True:
            # Build a list of candidate labels that have at least 2 samples remaining.
            candidate_labels = [label for label, inds in available.items() if len(inds) >= 2]

            if len(candidate_labels) < 5:
                print("Not enough labels to form a full batch")
                break
            
            # Randomly select 5 distinct labels
            chosen_labels = r.sample(candidate_labels, 5)
            
            batch = []
            # For each selected label, randomly sample 2 indices.
            for label in chosen_labels:
                inds = available[label]
                
                # Randomly select 2 indices for the current label
                pair = random.sample(inds, 2)
                batch.extend(pair)
                
                # Remove the selected indices from the available pool
                for i in pair:
                    available[label].remove(i)

            # Shuffle the batch order (optional)
            random.shuffle(batch)
            batches.append(batch)
    
        # Yield each batch.
        for batch in batches:
            yield batch

    def __len__(self):
        # Estimate the number of full batches by summing pairs available across labels
        total_pairs = sum(len(inds) // 2 for inds in self.label_to_indices.values())
        
        # Each batch requires 5 distinct pairs.
        return total_pairs // 5


