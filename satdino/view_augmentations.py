from copy import copy

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TVF
import numpy as np

from satdino import normalization_values, utils


class RandomResizedCrop(object):
    def __init__(self, size, scale, ratio=(3 / 4, 4 / 3), interpolation=Image.BICUBIC):
        self.transform = transforms.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=interpolation)

    def __call__(self, image, gsd):
        i, j, h, w = self.transform.get_params(image, self.transform.scale, self.transform.ratio)
        image = TVF.resized_crop(image, i, j, h, w, self.transform.size, self.transform.interpolation,
                                 antialias=self.transform.antialias)

        size = self.transform.size
        scale_factor_h = size[0] / h
        scale_factor_w = size[1] / w
        scale_factor = (scale_factor_w + scale_factor_h) / 2
        gsd = gsd / scale_factor

        return image, gsd

    
class DataAugmentationDINO(object):
    """
    Standard DINO augmentations.
    """
    def __init__(
            self,
            global_crops_scale,
            local_crops_scale,
            local_crops_number,
            normalization="imagenet",
            aspect_ratio=(3 / 4, 4 / 3)
    ):
        p_color_jitter = 0.8
        p_gray_scale = 0.2
        p_global_blur1 = 1.0
        p_global_blur2 = 0.1
        p_global_solar = 0.2
        p_local_blur = 0.5
        norm_mean, norm_std = normalization_values[normalization]

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=p_color_jitter
            ),
            transforms.RandomGrayscale(p=p_gray_scale),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC, ratio=aspect_ratio),
            flip_and_color_jitter,
            utils.GaussianBlur(p=p_global_blur1),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC, ratio=aspect_ratio),
            flip_and_color_jitter,
            utils.GaussianBlur(p_global_blur2),
            utils.Solarization(p_global_solar),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC, ratio=aspect_ratio),
            flip_and_color_jitter,
            utils.GaussianBlur(p=p_local_blur),
            normalize,
        ])

    def __call__(self, images, metadata=None):
        if not isinstance(images, list):
            images = [images]*3
        crops = []
        crops.append(self.global_transfo1(images[1]))
        crops.append(self.global_transfo2(images[2]))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(images[0]))

        if metadata is None:
            return crops
        return crops, metadata


class DataAugmentationDINO2(object):
    """
    Standard DINO augmentations, with metadata augmentations.
    """
    def __init__(
            self,
            global_crops_scale,
            local_crops_scale,
            local_crops_number,
            normalization="imagenet",
            aspect_ratio=(3 / 4, 4 / 3)
    ):
        p_color_jitter = 0.8
        p_gray_scale = 0.2
        p_global_blur1 = 1.0
        p_global_blur2 = 0.1
        p_global_solar = 0.2
        p_local_blur = 0.5
        norm_mean, norm_std = normalization_values[normalization]

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=p_color_jitter
            ),
            transforms.RandomGrayscale(p=p_gray_scale),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        # first global crop
        self.global_resize_transform = RandomResizedCrop(224, global_crops_scale, ratio=aspect_ratio, interpolation=Image.Resampling.BICUBIC)
        self.global_transfo1 = transforms.Compose([
            flip_and_color_jitter,
            utils.GaussianBlur(p=p_global_blur1),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            flip_and_color_jitter,
            utils.GaussianBlur(p_global_blur2),
            utils.Solarization(p_global_solar),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_resize_transform = RandomResizedCrop(96, local_crops_scale, ratio=aspect_ratio, interpolation=Image.Resampling.BICUBIC)
        self.local_transfo = transforms.Compose([
            flip_and_color_jitter,
            utils.GaussianBlur(p=p_local_blur),
            normalize,
        ])

    def __call__(self, images, metadata=None):
        crops = []
        _metadata = {}

        # copy metadata for global crops
        if metadata is not None:
            for k, v in metadata.items():
                _metadata[k] = copy(v[1:])

        # resize global crops (gsd changes)
        gsd1, gsd2 = _metadata['gsd']
        crop1, gsd1 = self.global_resize_transform(images[1], gsd1)
        crop2, gsd2 = self.global_resize_transform(images[2], gsd2)

        # apply global transforms
        crops.append(self.global_transfo1(crop1))
        crops.append(self.global_transfo2(crop2))
        _metadata['gsd'] = [gsd1, gsd2]

        # create local views
        for _ in range(self.local_crops_number):
            if metadata is not None:
                for k, v in metadata.items():
                    _metadata[k].append(copy(v[0]))

            # resize local crops
            gsd = metadata["gsd"][0]
            crop, gsd = self.local_resize_transform(images[0], gsd)

            # apply local transforms
            crops.append(self.local_transfo(crop))
            _metadata["gsd"][-1] = gsd

        return crops, _metadata


class DataAugmentationSatDINO(object):
    """
    SatDINO augmentations, with metadata augmentations. Local scales are sampled from uniform ranges.
    """
    def __init__(
            self,
            global_crops_scale,
            local_crops_scale,
            local_crops_number,
            normalization="imagenet",
            aspect_ratio=(3 / 4, 4 / 3)
    ):
        p_color_jitter = 0.8
        p_gray_scale = 0.2
        p_global_blur1 = 1.0
        p_global_blur2 = 0.1
        p_global_solar = 0.2
        p_local_blur = 0.5
        norm_mean, norm_std = normalization_values[normalization]

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=p_color_jitter
            ),
            transforms.RandomGrayscale(p=p_gray_scale),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        # first global crop
        self.global_resize_transform = RandomResizedCrop(224, global_crops_scale, ratio=aspect_ratio, interpolation=Image.Resampling.BICUBIC)
        self.global_transfo1 = transforms.Compose([
            flip_and_color_jitter,
            utils.GaussianBlur(p=p_global_blur1),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            flip_and_color_jitter,
            utils.GaussianBlur(p_global_blur2),
            utils.Solarization(p_global_solar),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        min_scale, max_scale = local_crops_scale
        step = (max_scale - min_scale) / local_crops_number
        bottom_scale = (np.arange(local_crops_number) * step) + min_scale 
        top_scale = bottom_scale + step
        local_scales = list(zip(bottom_scale, top_scale))
        self.local_resize_transform = []
        for local_scale in local_scales:
            _local_resize_transform = RandomResizedCrop(96, local_scale, ratio=aspect_ratio, interpolation=Image.Resampling.BICUBIC)
            self.local_resize_transform.append(_local_resize_transform)

        self.local_transfo = transforms.Compose([
            flip_and_color_jitter,
            utils.GaussianBlur(p=p_local_blur),
            normalize,
        ])

    def __call__(self, images, metadata=None):
        crops = []
        _metadata = {}

        # copy metadata for global crops
        if metadata is not None:
            for k, v in metadata.items():
                _metadata[k] = copy(v[1:])

        # resize global crops (gsd changes)
        gsd1, gsd2 = _metadata['gsd']
        crop1, gsd1 = self.global_resize_transform(images[1], gsd1)
        crop2, gsd2 = self.global_resize_transform(images[2], gsd2)

        # apply global transforms
        crops.append(self.global_transfo1(crop1))
        crops.append(self.global_transfo2(crop2))
        _metadata['gsd'] = [gsd1, gsd2]

        # create local views
        for i in range(self.local_crops_number):
            if metadata is not None:
                for k, v in metadata.items():
                    _metadata[k].append(copy(v[0]))

            # resize local crops
            gsd = metadata["gsd"][0]
            crop, gsd = self.local_resize_transform[i](images[0], gsd)

            # apply local transforms
            crops.append(self.local_transfo(crop))
            _metadata["gsd"][-1] = gsd

        return crops, _metadata

