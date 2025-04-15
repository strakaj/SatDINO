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
import sys

import numpy as np
import torch
from torch import nn
from torchvision import transforms as pth_transforms

from satdino import utils, normalization_values
from satdino.classification_dataset import ClassificationDataset
from satdino.train_utils import get_model_type


dataset_size = {
    "eurosat": 64,
    "resisc45": 256,
    "uc_merced": 256,
    "whu-rs19": 600,
    "rs_c11": 512,
    "siri-whu": 200,
    "fmow": 224
}

dataset_classes = {
    "eurosat": 10,
    "resisc45": 45,
    "uc_merced": 21,
    "whu-rs19": 19,
    "rs_c11": 11,
    "siri-whu": 12,
    "fmow": 63
}

class ReturnIndexDataset(ClassificationDataset):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx

    
def create_model(arch: str, patch_size: int, checkpoint_key: str = "teacher", pretrained_weights: str = "", model_type: str = ""):
    """Build model and load checkpoint."""
    if "vit" in arch:
        vit_models = get_model_type(model_type)
        model = vit_models.__dict__[arch](patch_size=patch_size, num_classes=0)
        print(f"Model {arch} {patch_size}x{patch_size} built.")
    else:
        print(f"Architecture {arch} non supported")
        sys.exit(1)
    model.cuda()

    if pretrained_weights:
        utils.load_pretrained_weights(model, pretrained_weights, checkpoint_key, arch, patch_size)
    model.eval()

    return model


def prepare_dataset(image_folder: str, scaled_image_size: int, num_workers: int = 4, batch_size: int = 4,
                    normalization="imagenet"):
    """Prepare dataloaders."""
    # prepare transforms
    ratio = 256 / 224
    transform = pth_transforms.Compose([
        pth_transforms.Resize(int(np.round(scaled_image_size * ratio).astype(int)), interpolation=3),
        pth_transforms.CenterCrop(scaled_image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(*normalization_values[normalization]),
    ])

    # create datasets
    root_folder = os.path.dirname(image_folder)
    train_names = os.path.join(root_folder, "train.txt")
    val_names = os.path.join(root_folder, "val.txt")

    dataset_train = ReturnIndexDataset(train_names, image_folder, transform)
    dataset_val = ReturnIndexDataset(val_names, image_folder, transform)
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
    val_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()

    return dataloader_train, dataloader_val, train_labels, val_labels


@torch.no_grad()
def extract_features(model, data_loader):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = []
    for samples, index in metric_logger.log_every(data_loader, 50):
        samples = samples.cuda()
        _features = model(samples).clone()
        features.append(_features)
    features = torch.cat(features, 0)
    return features


@torch.no_grad()
def knn_classifier(
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: torch.Tensor,
        val_labels: torch.Tensor,
        k: int,
        temperature: float,
        num_classes: int = 1000
):
    train_features = train_features.cuda()
    val_features = val_features.cuda()
    train_labels = train_labels.cuda()
    val_labels = val_labels.cuda()

    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = val_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = val_features[
                   idx: min((idx + imgs_per_chunk), num_test_images), :
                   ]
        targets = val_labels[idx: min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(temperature).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


def knn_main(
        model,
        dataset_folder: str,
        scales: list = [1.0],
        k: list = [20],
        num_workers: int = 4,
        batch_size: int = 64,
        temperature: float = 0.07,
        normalization: str = "imagenet",
        patch_size: int = 16,
        dataset_name:str = ""
):
    model.eval()

    results = {}
    for scale in scales:
        init_image_size = dataset_size[dataset_name]
        scaled_image_size = int(np.round(scale * init_image_size))
        src_scaled_image_size = scaled_image_size

        num_patches = src_scaled_image_size / patch_size
        if np.abs(num_patches - np.round(num_patches)) > 0.00001:
            _num_patches = np.max([1, np.floor(num_patches)])
            _scaled_image_size = _num_patches * patch_size

            print(f"Number of patches set from {num_patches} to {_num_patches}.")
            print(f"Setting size for scale: {scale}, from {src_scaled_image_size} to {_scaled_image_size}.")
            scaled_image_size = int(_scaled_image_size)

        if scaled_image_size < patch_size:
            print(f"Setting size for scale: {scale}, from {src_scaled_image_size} to {patch_size}.")
            scaled_image_size = patch_size

        results_key = f"{init_image_size}->{src_scaled_image_size}({scaled_image_size})"
        results[results_key] = []
            
        dataloader_train, dataloader_val, train_labels, val_labels = prepare_dataset(
            image_folder=dataset_folder,
            scaled_image_size=scaled_image_size,
            num_workers=num_workers,
            batch_size=batch_size,
            normalization=normalization
        )
            
        print("Extracting features for train set...")
        train_features = extract_features(model, dataloader_train)
        print("Extracting features for val set...")
        val_features = extract_features(model, dataloader_val)

        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        val_features = nn.functional.normalize(val_features, dim=1, p=2)

        for _k in k:
            top1, _ = knn_classifier(train_features, train_labels, val_features, val_labels, _k, temperature,
                                     dataset_classes[dataset_name])
            results[results_key].append(top1)
    model.train()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN.')
    parser.add_argument('--dataset_folder', type=str, nargs='+', help='Image folder of the dataset.')
    parser.add_argument("--output_file", type=str, default="", help="Output file")
    parser.add_argument('--scales', default=[1.0], type=float, nargs='+')
    parser.add_argument('--k', default=[20], type=int, help='Number of NN to use. 20 is usually working the best.',
                        nargs='+')
    parser.add_argument('--temperature', default=0.07, type=float, help='Temperature used in the voting coefficient')
    parser.add_argument('--normalization', default='imagenet', type=str,
                        help='What normalization (mean and std) to use for the dataset.')

    parser.add_argument('--model_type', default='dino', type=str, help='Model version - dino or satdino.')
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture of the model.')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")

    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()

    model = create_model(args.arch, args.patch_size, pretrained_weights=args.pretrained_weights, model_type=args.model_type)
    all_results = {}
    for dataset_folder in args.dataset_folder:
        dataset_name = os.path.basename(os.path.dirname(dataset_folder))
        normalization = dataset_name if args.normalization == "dataset" else args.normalization
        results = knn_main(
            model,
            dataset_folder=dataset_folder,
            scales=args.scales,
            k=args.k,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            temperature=args.temperature,
            normalization=normalization,
            dataset_name=dataset_name
        )

        # get averages across scales
        avgs = []
        for k_idx in range(len(args.k)):
            avg = float(np.mean([values[k_idx] for values in results.values()]))
            avgs.append(avg)
        results["average"] = avgs

        all_results[dataset_name] = results
        print(all_results)

    # save results
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'a') as f:
            f.write(json.dumps(all_results, indent=4) + "\n")
