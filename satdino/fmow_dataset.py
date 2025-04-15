import os
import json
from copy import copy

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

CATEGORIES = ["airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture", "archaeological_site",
              "barn", "border_checkpoint", "burial_site", "car_dealership", "construction_site", "crop_field", "dam",
              "debris_or_rubble", "educational_institution", "electric_substation", "factory_or_powerplant",
              "fire_station", "flooded_road", "fountain", "gas_station", "golf_course", "ground_transportation_station",
              "helipad", "hospital", "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse",
              "military_facility", "multi-unit_residential", "nuclear_powerplant", "office_building",
              "oil_or_gas_facility", "park", "parking_lot_or_garage", "place_of_worship", "police_station", "port",
              "prison", "race_track", "railway_bridge", "recreational_facility", "road_bridge", "runway", "shipyard",
              "shopping_mall", "single-unit_residential", "smokestack", "solar_farm", "space_facility", "stadium",
              "storage_tank", "surface_mine", "swimming_pool", "toll_booth", "tower", "tunnel_opening",
              "waste_disposal", "water_treatment_facility", "wind_farm", "zoo", ]


class FMoWRGBDataset(Dataset):
    def __init__(self, data_path, root_path="", transform=None, return_metadata=False):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        """
        self.transforms = transform
        self.return_metadata = return_metadata
        self.data = pd.read_csv(data_path)
        self.data["image_path"] = self.data["image_path"].apply(lambda p: os.path.join(root_path, p.strip("/")))
        
        self.image_paths = np.asarray([p for p in self.data["image_path"]])
        self.labels = np.asarray(self.data["category_idx"])
        if return_metadata:
            self.metadata_keys = ['gsd', 'hour', 'month', 'year']
            self.metadata = self.data[self.metadata_keys]
        self.samples = list(zip(self.image_paths, self.labels))
        self.data_len = len(self.data)
        
    @staticmethod
    def load_image(path):
        image = Image.open(path)
        return image
    
    @staticmethod
    def load_json(path):
        with open(path, "r") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self.load_image(image_path)
        
        if self.transforms is not None:
            image = self.transforms(image)
            
        image_label = self.labels[index]
        if self.return_metadata:
            metadata = {}
            for idx, val in enumerate(self.metadata.iloc[index]):
                val = int(val) if abs(val-int(val)) < 0.0001 else val
                metadata[self.metadata_keys[idx]] = val
            return image, image_label, metadata
        return image, image_label
    

class FMoWRGBTemporalDataset(FMoWRGBDataset):
    def __init__(self, *args, return_temporal=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_temporal = return_temporal
        self.image_data = {}
        self.labels = []

        for pid in tqdm(self.data["id"].unique(), desc="Preparing data"):
            idata = self.data[self.data["id"]==pid]
            indexes = list(idata.index)
            if len(indexes) == 1:
                self.image_data[indexes[0]] = indexes
                self.labels.append(self.data.iloc[indexes[0]]["category_idx"])
            else:
                for index in indexes:
                    _indexes = copy(indexes)
                    _indexes.remove(index)
                    self.image_data[index] = _indexes
                    self.labels.append(self.data.iloc[index]["category_idx"])
            
        self.samples = list(zip(self.image_data, self.labels))
        self.data_len = len(self.data)

    def __getitem__(self, index):
        if self.return_temporal:
            temporal_indexes = self.image_data[index]
            glob_data = self.data.iloc[[index]]
            local_global_data = self.data.iloc[temporal_indexes].sample(2, replace=True)
            image_data = pd.concat([local_global_data, glob_data])
        else:
            image_data = self.data.iloc[[index]].sample(3, replace=True)

        images = []
        if self.return_temporal:
            for _, data in image_data.iterrows():
                image = self.load_image(data["image_path"])
                images.append(image)
        else:
            image = self.load_image(image_data["image_path"].iloc[0])
            images = [image]*len(image_data)

        image_label = self.labels[index]
        metadata = {name: list(image_data[name]) for name in ["year", "month", "hour", "gsd"]}
        
        if self.transforms is not None:
            images, metadata = self.transforms(images, metadata)
            
        return images, image_label, metadata
