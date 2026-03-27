# adapted from https://github.com/NVIDIA/OpenSeq2Seq/blob/master/run.py

import os
import ast
import copy
import runpy
import random
import argparse
import numpy as np
import shutil
import time

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel, DataParallel

from openchem.models.openchem_model import build_training, fit, evaluate, predict

from openchem.data.utils import create_loader
from openchem.utils.utils import get_latest_checkpoint, deco_print
from openchem.utils.utils import flatten_dict, nested_update, nest_dict
from openchem.utils import comm
from openchem.utils.textlogger import setup_textlogger
from openchem.utils import metrics
from openchem.data.utils import sanitize_smiles


def main():
    parser = argparse.ArgumentParser(description='Experiment parameters')
    parser.add_argument("--use_cuda", default=torch.cuda.is_available(),
                        help="Whether to train on GPU")
    parser.add_argument("--config_file", required=True,
                        help="Path to the configuration file")
    parser.add_argument("--mode", default='train',
                        help="Could be \"train\", \"eval\", \"train_eval\", \"predict\"")
    parser.add_argument('--continue_learning', dest='continue_learning',
                        action='store_true',
                        help="whether to continue learning")
    parser.add_argument("--force_checkpoint", dest="force_checkpoint",
                        default="",
                        help="Full path to a pretrained snapshot "
                             "(e.g. useful for knowledge transfer or)")
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--random_seed', default=0, type=int, metavar='N',
                        help='random_seed (default: 0)')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--copy_config_file", action="store_true",
                        help="Copy config file to logdir (useful in training)")

    args, unknown = parser.parse_known_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) \
        if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.dist_backend,
                                init_method='env://')
        print(f'Distributed process with rank {args.local_rank:d} initalized')

    cudnn.benchmark = True

    if args.mode not in ['train', 'eval', 'train_eval', 'infer', 'predict']:
        raise ValueError("Mode has to be one of "
                         "['train', 'eval', 'train_eval', 'infer', 'predict']")
    config_module = runpy.run_path(args.config_file)

    model_config = config_module.get('model_params', None)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    model_config['use_cuda'] = args.use_cuda
    if model_config is None:
        raise ValueError('model_params dictionary has to be '
                         'defined in the config file')
    model_object = config_module.get('model', None)
    if model_object is None:
        raise ValueError('model class has to be defined in the config file')

        # after we read the config, trying to overwrite some of the properties
        # with command line arguments that were passed to the script
    parser_unk = argparse.ArgumentParser()
    for pm, value in flatten_dict(model_config).items():
        if isinstance(value, bool):
            parser_unk.add_argument('--' + pm, default=value,
                                    type=ast.literal_eval)
        elif isinstance(value, (int, float, str)):
            parser_unk.add_argument('--' + pm, default=value, type=type(value))

    config_update = parser_unk.parse_args(unknown)
    nested_update(model_config, nest_dict(vars(config_update)))

    # checking that everything is correct with log directory
    logdir = model_config['logdir']
    ckpt_dir = os.path.join(logdir, 'checkpoint')

    if args.force_checkpoint:
        assert not args.continue_learning, \
            "force_checkpoint and continue_learning are " \
            "mutually exclusive flags"
        checkpoint = args.force_checkpoint
        assert os.path.isfile(checkpoint), f"{checkpoint} is not a file"
        cur_epoch = 0
    elif args.mode in ['eval', 'infer', 'predict'] or args.continue_learning:
        checkpoint = get_latest_checkpoint(ckpt_dir)
        if checkpoint is None:
            raise OSError(
                f"Failed to find model checkpoint under "
                f"{ckpt_dir}. Can't load the model"
            )
        cur_epoch = int(os.path.basename(checkpoint).split("_")[-1]) + 1
    else:
        checkpoint = None
        cur_epoch = 0

    if not os.path.exists(logdir):
        comm.mkdir(logdir)
        print(f'Directory {logdir} created')
    elif os.path.isfile(logdir):
        raise OSError(
            "There is a file with the same name as \"logdir\" "
            "parameter. You should change the log directory path "
            "or delete the file to continue.")

    if not os.path.exists(ckpt_dir):
        comm.mkdir(ckpt_dir)
        print(f'Directory {ckpt_dir} created')
    elif os.path.isdir(ckpt_dir) and os.listdir(ckpt_dir) != []:
        if not args.continue_learning and args.mode not in ['eval', 'infer', 'predict']:
            raise OSError(
                "Log directory is not empty. If you want to "
                "continue learning, you should provide "
                "\"--continue_learning\" flag")

    doprint = comm.is_main_process()
    tofile = os.path.join(logdir, "log.txt")
    logger = setup_textlogger("openchem", doprint, tofile)
    msg = "Running with config:\n"
    for k, v in sorted(flatten_dict(model_config).items()):
        msg += (f"{k}:\t{v}\n").expandtabs(50)
    logger.info(f"Running on {comm.get_world_size():d} GPUs")
    logger.info(f"Logging directory is set to {logdir}")
    logger.info(msg)
    if args.copy_config_file:
        shutil.copy(args.config_file, logdir)

    train_config = copy.deepcopy(model_config)
    eval_config = copy.deepcopy(model_config)

    if args.mode == 'train' or args.mode == 'train_eval':
        if 'train_params' in config_module:
            nested_update(train_config,
                          copy.deepcopy(config_module['train_params']))
    if args.mode in ['eval', 'train_eval', 'infer', 'predict']:
        if 'eval_params' in config_module:
            nested_update(eval_config,
                          copy.deepcopy(config_module['eval_params']))

    if args.mode == "train" or args.mode == "train_eval":
        train_dataset = copy.deepcopy(model_config['train_data_layer'])
        if model_config['task'] == 'classification':
            train_dataset.target = train_dataset.target.reshape(-1)
        if args.distributed:
            train_sampler = DistributedSampler(train_dataset)
        else:
            train_sampler = None
        train_loader = create_loader(train_dataset,
                                     batch_size=model_config['batch_size'],
                                     shuffle=(train_sampler is None),
                                     num_workers=args.workers,
                                     pin_memory=True,
                                     sampler=train_sampler)
    else:
        train_loader = None

    if args.mode == "predict" and ('predict_data_layer' not in model_config.keys()
                                   or model_config['predict_data_layer'] is None):
        raise OSError(
            "When model is run in 'predict' mode, "
            "prediction data layer must be specified")

    if args.mode == "predict":
        predict_dataset = copy.deepcopy(model_config['predict_data_layer'])
        predict_loader = create_loader(predict_dataset,
                                       batch_size=model_config['batch_size'],
                                       shuffle=False,
                                       num_workers=1,
                                       pin_memory=True)
    else:
        predict_loader = None

    if args.mode in ["eval", "train_eval"] and (
            'val_data_layer' not in model_config.keys()
            or model_config['val_data_layer'] is None):
        raise OSError(
            "When model is run in 'eval' or 'train_eval' modes, "
            "validation data layer must be specified")

    if args.mode in ["eval", "train_eval"]:
        val_dataset = copy.deepcopy(model_config['val_data_layer'])
        if model_config['task'] == 'classification':
            val_dataset.target = val_dataset.target.reshape(-1)
        val_loader = create_loader(val_dataset,
                                   batch_size=model_config['batch_size'],
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True)
    else:
        val_loader = None

    model_config['train_loader'] = train_loader
    model_config['val_loader'] = val_loader
    model_config['predict_loader'] = predict_loader

    # create model
    model = model_object(params=model_config)

    if args.use_cuda:
        model = model.to('cuda')

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank],
                                        output_device=args.local_rank)
    else:
        model = DataParallel(model)

    if checkpoint is not None:
        logger.info(f"Loading model from {checkpoint}")
        weights = torch.load(checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(weights)
    else:
        logger.info("Starting training from scratch")
    if args.mode in ["train", "train_eval"]:
        logger.info(f"Training is set up from epoch {cur_epoch:d}")

    criterion, optimizer, lr_scheduler = build_training(model, model_config)

    if args.mode == 'train':
        fit(model, lr_scheduler, train_loader, optimizer, criterion,
            model_config, eval=False, cur_epoch=cur_epoch)
    elif args.mode == 'train_eval':
        fit(model, lr_scheduler, train_loader, optimizer, criterion,
            model_config, eval=True, val_loader=val_loader, cur_epoch=cur_epoch)
    elif args.mode == "eval":
        evaluate(model, val_loader, criterion)
    elif args.mode == "predict":
        predict(model, predict_loader)
    elif args.mode == "infer":
        comm.synchronize()
        start_time = time.time()

        #if comm.get_world_size() > 1:
        #    seed = comm.get_rank() * 10000
        #    random.seed(seed)
        #    np.random.seed(seed)
        #    torch.manual_seed(seed)
        #    torch.cuda.manual_seed_all(seed)

        model.eval()
        smiles = []

        with torch.no_grad():
            for i in range(1):
                batch_smiles = model(None, batch_size=1024)
                smiles.extend(batch_smiles)
                print(f"Iteration {i+1:d}: {len(batch_smiles):d} smiles")

        if comm.get_world_size() > 1:
            path = os.path.join(logdir, f"debug_smiles_{comm.get_rank():d}.txt")
            with open(path, "w") as f:
                for s in smiles:
                    f.write(s + "\n")

            comm.synchronize()

            if not comm.is_main_process():
                return

            smiles = []
            for i in range(comm.get_world_size()):
                path = os.path.join(logdir, f"debug_smiles_{i:d}.txt")
                with open(path) as f:
                    smiles_local = f.readlines()
                os.remove(path)

                smiles_local = [s.rstrip() for s in smiles_local]
                smiles.extend(smiles_local)

        path = os.path.join(logdir, "debug_smiles.txt")
        with open(path, "w") as f:
            for s in smiles:
                f.write(s + "\n")

        print(f"Generated {len(smiles):d} molecules in {time.time() - start_time:.1f} seconds")

        eval_metrics = model_config['eval_metrics']
        score = eval_metrics(None, smiles)
        qed_score = metrics.qed(smiles)
        logger.info(f"Eval metrics = {score:.2f}")
        logger.info(f"QED score = {qed_score:.2f}")

        smiles, idx = sanitize_smiles(
            smiles,
            logging="info"
        )


if __name__ == '__main__':
    main()

