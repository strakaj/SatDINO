import json
from tqdm import tqdm
import pandas as pd
import os

category_names = ['false_detection', 'airport', 'airport_hangar', 'airport_terminal', 'amusement_park',
                               'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint', 'burial_site',
                               'car_dealership', 'construction_site', 'crop_field', 'dam', 'debris_or_rubble',
                               'educational_institution', 'electric_substation', 'factory_or_powerplant',
                               'fire_station', 'flooded_road', 'fountain', 'gas_station', 'golf_course',
                               'ground_transportation_station', 'helipad', 'hospital', 'interchange', 'lake_or_pond',
                               'lighthouse', 'military_facility', 'multi-unit_residential', 'nuclear_powerplant',
                               'office_building', 'oil_or_gas_facility', 'park', 'parking_lot_or_garage',
                               'place_of_worship', 'police_station', 'port', 'prison', 'race_track', 'railway_bridge',
                               'recreational_facility', 'impoverished_settlement', 'road_bridge', 'runway', 'shipyard',
                               'shopping_mall', 'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility',
                               'stadium', 'storage_tank', 'surface_mine', 'swimming_pool', 'toll_booth', 'tower',
                               'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def data_to_csv(struct_data_file: str, output_csv_file: str, split_name=None):
    with open(struct_data_file, "r") as f:
        data = json.load(f)

    new_data = {
        "id": [],
        "image_path": [],
        "feature_path": [],
        "split": [],
        "category_name": [],
        "category_idx": [],
        "gsd": [],
        "hour": [],
        "month": [],
        "year": [],
    }

    for d in tqdm(data, desc=f"Saving: {split_name}"):
        path = d["img_path"].split(os.sep)
        split, category = (path[-5], path[-4])
        pid = "/".join(path[-4:-2])

        if split != split_name:
            continue

        # parse feature data
        features = load_json(d["features_path"])
        gsd = features[0]
        year = int(features[4])
        month = int(features[5] * 12)
        hour = int(features[7])

        # create relative paths
        new_features_path = d["features_path"].split("/")
        new_features_path = "/".join(new_features_path[-7:])
        new_img_path = d["img_path"].split("/")
        new_img_path = "/".join(new_img_path[-7:])

        # store data
        new_data["id"].append(pid)
        new_data["image_path"].append(new_img_path)
        new_data["feature_path"].append(new_features_path)
        new_data["split"].append(split)
        new_data["category_name"].append(category)
        new_data["category_idx"].append(category_names.index(category))
        new_data["gsd"].append(gsd)
        new_data["hour"].append(hour)
        new_data["month"].append(month)
        new_data["year"].append(year)

    new_data = pd.DataFrame(new_data)
    new_data.to_csv(output_csv_file, index=False)

    return new_data
