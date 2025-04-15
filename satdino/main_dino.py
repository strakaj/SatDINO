# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb

from satdino import utils
from satdino.eval_knn import knn_main
from satdino.vision_transformer import DINOHead
from satdino.train_utils import get_augmentation_type, get_dataset_type, get_model_type


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_large', 'vit_tiny', 'vit_small', 'vit_base'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.07, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument('--pos_encoding', default='learnable', type=str,
        choices=['sin_cos', 'learnable', 'none'], help="""Method of position encoding.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.25, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.25),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/train.csv', type=str,
        help='Path to the training data (CSV file) for the fMoW dataset.')
    parser.add_argument('--data_root', default='', type=str,
        help='Path to the training images for the fMoW dataset.')
    parser.add_argument('--init_model_path', default="", type=str,
        help="Path to a pre-trained model from which to initialize the training process. Leave empty to train from scratch.")
    parser.add_argument('--knn_dataset_folder', type=str, nargs='+',
        help='Path(s) to one or more image folders containing the k-NN dataset for evaluation.')
    parser.add_argument('--output_folder', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--use_xformers', type=utils.bool_flag, default=False)
    
    # Wandb parameters
    parser.add_argument('--wandb_api_key', default='', type=str)
    parser.add_argument('--entity', type=str)
    parser.add_argument('--project', default='', type=str)
    parser.add_argument('--group', type=str, help='Name of wandb group.')
    parser.add_argument('--name', default='', type=str, help='Name of wandb experiment.')
    parser.add_argument('--tags', nargs='*', type=str)

    # Dataset options
    parser.add_argument('--dataset_type', default="basic", type=str, choices=["basic", "temporal"],
        help="""Type of dataset to use for training:
                - 'basic': Each data sample consists of a single image.
                - 'temporal': Each data sample consists of an image and two additional random images from the same temporal sequence.   
                The 'temporal' dataset can also be switched to a non-temporal version, 
                where three identical images will be returned.""")
    parser.add_argument('--temporal_dataset', type=utils.bool_flag, default=False,
        help="""If using the temporal dataset, this flag determines whether to use the temporal or non-temporal version:
                - True: Use the temporal version (three images from the same temporal sequence).
                - False: Use the non-temporal version (three identical images).""")

    # Transform options
    parser.add_argument('--augmentation_type', default="dino", type=str, choices=["dino", "dino2", "satdino"],
        help="""Choose the type of data augmentation to apply during training:
                - 'dino': The original augmentation method used in DINO.
                - 'dino2': A modified version of DINO that also augments metadata (e.g., GSD).
                - 'satdino': Samples local views from uniform ranges.""")
    parser.add_argument('--aspect_ratio', default=[3 / 4, 4 / 3], nargs='+', type=float,
        help='View augmentations uses RandomResizedCrop which takes as argument aspect_ratio.')
    parser.add_argument('--normalization', default="imagenet", type=str, choices=["imagenet", "fmow"],
        help='Normalization values used in transforms.')

    return parser


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    augmentation_type = get_augmentation_type(args.augmentation_type)
    transform = augmentation_type(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        normalization=args.normalization,
        aspect_ratio=args.aspect_ratio,
    )

    dataset_type = get_dataset_type(args.dataset_type)
    kwargs = {}
    if args.dataset_type == "temporal":
        kwargs = {"return_temporal": args.temporal_dataset}
    dataset = dataset_type(args.data_path, args.data_root, transform=transform, **kwargs)

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    vit_models = get_model_type("dino")
    student = vit_models.__dict__[args.arch](
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path_rate,  # stochastic depth
        use_xformers=args.use_xformers,
        pos_encoding_method=args.pos_encoding,
    )
    teacher = vit_models.__dict__[args.arch](
        patch_size=args.patch_size,
        use_xformers=args.use_xformers,
        pos_encoding_method=args.pos_encoding,
    )
    embed_dim = student.embed_dim

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropTemporalWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropTemporalWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )

    # initialize model from pretrained weights
    if args.init_model_path:
        checkpoint = torch.load(args.init_model_path)
        if checkpoint["teacher"]['backbone.pos_embed'].shape[-1] != student.backbone.pos_embed.shape[-1]:
            print("Missmatch in position embedding -> removing position embeding from checkpoint")
            del checkpoint["teacher"]["backbone.pos_embed"]
        res = student.load_state_dict(checkpoint["teacher"], strict=False)
        print(res)
        print(f"Student initialized from: {args.init_model_path}.")
    
    # move models to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_folder, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)
        
        # ============ evaluate knn ... ============
        eval_stats = eval_epoch_knn(student, teacher, args.knn_dataset_folder, args.num_workers)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_folder, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_folder, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        
        if wandb.run is not None:
            epoch_stats = {f"epoch/{k}":v for k, v in train_stats.items()}
            epoch_stats.update({f"eval/{k}":v for k, v in eval_stats.items()})
            epoch_stats["epoch/step"] = epoch
            wandb.log(epoch_stats)
        
        if utils.is_main_process():
            with (Path(args.output_folder) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def eval_epoch_knn(student: nn.Module, teacher: nn.Module, dataset_folders: list, num_workers: int):
    eval_stats = {}
    if dataset_folders is None:
        return eval_stats

    for dataset_folder in dataset_folders:
        dataset_name = os.path.basename(os.path.dirname(dataset_folder))

        start_time_knn = time.time()
        teacher_results = knn_main(
            teacher.backbone, dataset_folder=dataset_folder, scales=[0.5], k=[20],
            num_workers=num_workers, normalization="dataset", dataset_name=dataset_name
        )
        student_results = knn_main(
            student.module.backbone, dataset_folder=dataset_folder, scales=[0.5], k=[20],
            num_workers=num_workers, normalization="dataset", dataset_name=dataset_name
        )
        teacher_results = list(teacher_results.values())[0][0]
        student_results = list(student_results.values())[0][0]
        eval_stats[f"knn_teacher_{dataset_name}"] = teacher_results
        eval_stats[f"knn_student_{dataset_name}"] = student_results
        print(dataset_name)
        print(f"    knn time: {time.time() - start_time_knn:.3f} s")
        print(f"    teacher: {teacher_results} acc")
        print(f"    student: {student_results} acc")

    return eval_stats


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, data in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = data[0]

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        
        if wandb.run is not None:
            wandb.log({
                'loss': loss.item(), 
                'lr': optimizer.param_groups[0]["lr"],
                'weight_decay': optimizer.param_groups[0]["weight_decay"],
            }, it)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm-up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def init_wandb(args):    
    args = vars(args)
    if args["wandb_api_key"]:
        os.environ['WANDB_API_KEY'] = args["wandb_api_key"]
    del args["wandb_api_key"]

    if args["project"] and os.environ['WANDB_API_KEY']:
        # initialize wandb
        kwarg_names = ["group", "name", "entity", "tags"]
        wandb_kwargs = {n: args[n] for n in kwarg_names if n in args and n}

        wandb.init(
            project=args["project"],
            config=args,
            **wandb_kwargs
        )
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
        

    # get local rank
    local_rank = 0
    if 'SLURM_PROCID' in os.environ:
        local_rank = int(os.environ['SLURM_LOCALID'])
        
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])

    # save config and init wandb
    if local_rank == 0:
        if args.output_folder:
            # create output folder
            experiment_folder = os.path.join(args.output_folder, args.name)
            os.makedirs(experiment_folder, exist_ok=True)
            args.output_folder = experiment_folder

            # save experiment data
            experiment_file = os.path.join(args.output_folder, "experiment_data.json")
            experiment_data = {}
            if os.path.exists(experiment_file):
                with open(experiment_file, "r") as f:
                    experiment_data = json.load(f)
            experiment_data['args'] = vars(args)
            with open(experiment_file, "w") as f:
                json.dump(experiment_data, f, indent=4)

        init_wandb(args)
        
    if wandb.run is not None:
        wandb.define_metric("epoch/step")
        wandb.define_metric("epoch/*", step_metric="epoch/step")
        wandb.define_metric("eval/*", step_metric="epoch/step")

    train_dino(args)
