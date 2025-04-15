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
import json
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import wandb
from torch import nn
from torchvision import transforms as pth_transforms

from satdino import utils, normalization_values
from satdino.train_utils import get_dataset_type, get_model_type


def eval_linear(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============

    vit_models = get_model_type(args.model_type)
    model = vit_models.__dict__[args.arch](
        patch_size=args.patch_size,
        num_classes=0,
        use_xformers=args.use_xformers,
        pos_encoding_method=args.pos_encoding,
    )
    embed_dim = model.embed_dim

    model.cuda()
    model.eval()

    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # load weights to evaluate
    if args.pretrained_weights and not args.evaluate:
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(*normalization_values[args.normalization]),
    ])

    dataset_type = get_dataset_type("basic")
    dataset_val = dataset_type(args.val_data_path, args.data_root, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    if args.evaluate:
        checkpoint = torch.load(args.pretrained_weights, map_location="cuda", weights_only=False)
        model.load_state_dict(checkpoint['teacher'], strict=True)
        linear_classifier.load_state_dict(checkpoint['linear_head'], strict=True)
        # utils.load_pretrained_linear_weights(linear_classifier, args.arch, args.patch_size)
        test_stats = validate_network(val_loader, model, linear_classifier)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(*normalization_values[args.normalization]),
    ])

    dataset_train = dataset_type(args.train_data_path, args.data_root, transform=train_transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # prepare optimizer
    params = None
    if args.finetune_mode == "head":
        params = linear_classifier.parameters()
    elif args.finetune_mode == "full":
        print(f"Fine tune mode: {args.finetune_mode}")
        if args.lr_head > 0:
            params = [
                {"params": model.parameters()},
                {"params": linear_classifier.parameters(), "lr": args.lr_head * (args.batch_size_per_gpu * utils.get_world_size()) / 256.},
            ]
        else:
            params = list(linear_classifier.parameters()) + list(model.parameters())

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
            weight_decay=0, # we do not apply weight decay
        )  
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
            momentum=0.9,
            weight_decay=0, # we do not apply weight decay
        )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    if args.output_folder:
        utils.restart_from_checkpoint(
            os.path.join(args.output_folder, "checkpoint.pth.tar"),
            run_variables=to_restore,
            state_dict=linear_classifier,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if args.finetune_mode == "full":
            chkp_path = os.path.join(args.output_folder, "checkpoint.pth.tar")
            if os.path.isfile(chkp_path):
                checkpoint = torch.load(chkp_path, map_location="cpu")
                msg = model.load_state_dict(checkpoint["model"], strict=False)
                print("=> loaded '{}' with msg {}".format("model", msg))
        
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
    
        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.finetune_mode)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, model, linear_classifier)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
            
            if wandb.run is not None:
                epoch_stats = {f"epoch/{k}":v for k, v in log_stats.items() if k != "epoch"}
                epoch_stats["epoch/step"] = log_stats["epoch"]
                wandb.log(epoch_stats)
            
        if utils.is_main_process() and args.output_folder:
            with (Path(args.output_folder) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            if args.finetune_mode == "full":
                save_dict["model"] = model.state_dict()
            torch.save(save_dict, os.path.join(args.output_folder, "checkpoint.pth.tar"))
            
            # save experiment data
            experiment_file = os.path.join(os.path.dirname(args.output_folder), "experiment_data.json")
            with open(experiment_file, "r") as f:
                experiment_data = json.load(f)
            if f'results_{args.finetune_mode}' not in experiment_data:
                experiment_data[f'results_{args.finetune_mode}'] = {}
            experiment_data[f'results_{args.finetune_mode}'][epoch] = test_stats
            with open(experiment_file, "w") as f:
                json.dump(experiment_data, f, indent=4)
            
            
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(model, linear_classifier, optimizer, loader, epoch, finetune_mode):
    linear_classifier.train()
    if finetune_mode == "full":
        model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for it, data in enumerate(metric_logger.log_every(loader, 20, header)):
        try:
            images, target = data
            it = len(loader) * epoch + it

            # move to gpu
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # forward
            if finetune_mode == "head":
                with torch.no_grad():
                    output = model(images)
            elif finetune_mode == "full":
                output = model(images)
            output = linear_classifier(output)

            # compute cross entropy loss
            loss = nn.CrossEntropyLoss()(output, target)

            # compute the gradients
            optimizer.zero_grad()
            loss.backward()

            # step
            optimizer.step()

            # log 
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            if wandb.run is not None:
                wandb.log({
                    'loss': loss.item(), 
                    'lr': optimizer.param_groups[0]["lr"],
                }, it)
        except Exception as e:
            print(f"Train iteration {it} failed!!!")
            print(e)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier):
    linear_classifier.eval()
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for data in metric_logger.log_every(val_loader, 20, header):
        try:
            images, target = data

            # move to gpu
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # forward
            with torch.no_grad():
                output = model(images)
                output = linear_classifier(output)
            loss = nn.CrossEntropyLoss()(output, target)

            if linear_classifier.module.num_labels >= 5:
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            else:
                acc1, = utils.accuracy(output, target, topk=(1,))

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            if linear_classifier.module.num_labels >= 5:
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        except Exception as e:
            print(f"Validation failed!!!")
            print(e)
    if linear_classifier.module.num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


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
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument("--lr_head", default=0, type=float)
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--val_data_path', default='/path/to/val.csv', type=str,
                        help='Path to the training data (CSV file) for the fMoW dataset.')
    parser.add_argument('--train_data_path', default='/path/to/train.csv', type=str,
                        help='Path to the training data (CSV file) for the fMoW dataset.')
    parser.add_argument('--data_root', default='', type=str,
                        help='Path to the training images for the fMoW dataset.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_folder', default="", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd'], help="""Type of optimizer. We recommend using adamw with ViTs.""")

    parser.add_argument('--normalization', default="imagenet", type=str, choices=["imagenet", "fmow"],
                        help='Normalization values used in transforms.')
    parser.add_argument('--use_xformers', type=utils.bool_flag, default=False)
    parser.add_argument('--pos_encoding', default='learnable', type=str, choices=['sin_cos', 'learnable'], help="Method of position encoding.")
    parser.add_argument('--finetune_mode', default="head", type=str, choices=["head", "full"], help="Only head training or full model?")
    parser.add_argument('--model_type', default="dino", type=str, choices=["dino", "satdino"])
    
    # Wandb parameters
    parser.add_argument('--wandb_api_key', default="", type=str)
    parser.add_argument('--entity', type=str)
    parser.add_argument('--project', default="", type=str)
    parser.add_argument('--group', type=str, help="Name of group in wandb.")
    parser.add_argument('--name', default="", type=str, help="Name of experiment in wandb.")
    parser.add_argument('--tags', nargs='*', type=str)

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
            args.output_folder = os.path.join(args.output_folder, f"finetune_{args.finetune_mode}")
            os.makedirs(args.output_folder, exist_ok=True)

            # save experiment data
            experiment_file = os.path.join(os.path.dirname(args.output_folder), "experiment_data.json")
            experiment_data = {}
            if os.path.exists(experiment_file):
                with open(experiment_file, "r") as f:
                    experiment_data = json.load(f)
            experiment_data[f'args_{args.finetune_mode}'] = vars(args)
            with open(experiment_file, "w") as f:
                json.dump(experiment_data, f, indent=4)

        init_wandb(args)

    if wandb.run is not None:
        wandb.define_metric("epoch/step")
        wandb.define_metric("epoch/*", step_metric="epoch/step")
        wandb.define_metric("eval/*", step_metric="epoch/step")

    eval_linear(args)
