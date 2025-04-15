from satdino.fmow_dataset import FMoWRGBDataset, FMoWRGBTemporalDataset
from satdino.view_augmentations import DataAugmentationDINO, DataAugmentationDINO2, DataAugmentationSatDINO
import satdino.vision_transformer as vits_dino
import satdino.vision_transformer_satdino as vits_satdino

_model_options = {
    "dino": vits_dino,
    "satdino": vits_satdino
}

_dataset_options = {
    "basic": FMoWRGBDataset,
    "temporal": FMoWRGBTemporalDataset
}

_augmentation_options = {
    "dino": DataAugmentationDINO,
    "dino2": DataAugmentationDINO2,
    "satdino": DataAugmentationSatDINO
}


def get_model_type(model_type):
    return _model_options[model_type]

def get_dataset_type(dataset_type):
    return _dataset_options[dataset_type]

def get_augmentation_type(augmentation_type):
    return _augmentation_options[augmentation_type]


