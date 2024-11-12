import time
import torch

from utils.logger import *
from utils.config import *
from tools import svm_run_net as svm
from utils import parser, dist_utils, misc
from tools import test_run_net as test_net
from tools import pretrain_run_net as pretrain
from tools import finetune_run_net as finetune
from tools import zeroshot_run_net as zeroshot


def main():
    # args
    args = parser.get_args()
    # CUDA
    torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.distributed:
        dist_utils.init_dist(args.local_rank)
        args.world_size = torch.distributed.get_world_size()

    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)

    # config
    config = get_config(args, logger=logger)
    # batch size
    dist_utils.set_batch_size(args, config)
    # log
    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank,
                             deterministic=args.deterministic)  # seed + rank, for augmentation

    if args.shot != -1:
        config.dataset.train.others.shot = args.shot
        config.dataset.train.others.way = args.way
        config.dataset.train.others.fold = args.fold
        config.dataset.val.others.shot = args.shot
        config.dataset.val.others.way = args.way
        config.dataset.val.others.fold = args.fold

    # run
    if args.test:
        test_net(args, config)
    elif args.zeroshot:
        zeroshot(args, config)
    elif args.svm:
        svm(args, config)
    elif args.finetune_model:
        finetune(args, config)
    else:
        pretrain(args, config)


if __name__ == '__main__':
    main()
