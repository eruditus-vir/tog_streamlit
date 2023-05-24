from enum import Enum
import pathlib
import os
import gdown
from typing import List
import logging


class ModelName(Enum):
    YOLOv3ABO = "Yolov3Abo"
    YOLOv3SEA = "Yolov3Sea"
    YOLOv3ABOSEA = "Yolov3AboSea"
    YOLOv5ABO = "Yolov5Abo"
    YOLOv5SEA = "Yolov5Sea"
    YOLOv5ABOSEA = "Yolov5AboSea"
    YOLOv8ABO = "Yolov8Abo"
    YOLOv8SEA = "Yolov8Sea"
    YOLOv8ABOSEA = "Yolov8AboSea"

    @staticmethod
    def get_all_model_names_in_str() -> List[str]:
        return [
            ModelName.YOLOv3ABO.value,
            ModelName.YOLOv3SEA.value,
            ModelName.YOLOv3ABOSEA.value,
            ModelName.YOLOv5ABO.value,
            ModelName.YOLOv5SEA.value,
            ModelName.YOLOv5ABOSEA.value,
            ModelName.YOLOv8ABO.value,
            ModelName.YOLOv8SEA.value,
            ModelName.YOLOv8ABOSEA.value
        ]

    @classmethod
    def from_str(cls, model_name: str):
        model_name_dict_internal = {
            'Yolov3Abo': ModelName.YOLOv3ABO,
            'Yolov3Sea': ModelName.YOLOv3SEA,
            'Yolov3AboSea': ModelName.YOLOv3ABOSEA,
            'Yolov5Abo': ModelName.YOLOv5ABO,
            'Yolov5Sea': ModelName.YOLOv5SEA,
            'Yolov5AboSea': ModelName.YOLOv5ABOSEA,
            'Yolov8Abo': ModelName.YOLOv8ABO,
            'Yolov8Sea': ModelName.YOLOv8SEA,
            'Yolov8AboSea': ModelName.YOLOv8ABOSEA
        }
        if model_name not in model_name_dict_internal.keys():
            raise Exception('{} is not acceptable'.format(model_name))
        return model_name_dict_internal[model_name]


model_names_to_file_names = {
    ModelName.YOLOv3ABO: 'yolov3_abo_single.pt',
    ModelName.YOLOv3SEA: 'yolov3_sea_single.pt',
    ModelName.YOLOv3ABOSEA: 'yolov3_abo_sea_single.pt',
    ModelName.YOLOv5ABO: 'yolov5_abo_single.pt',
    ModelName.YOLOv5SEA: 'yolov5_sea_single.pt',
    ModelName.YOLOv5ABOSEA: 'yolov5_abo_sea_single.pt',
    ModelName.YOLOv8ABO: 'yolov8_abo_single.pt',
    ModelName.YOLOv8SEA: 'yolov8_sea_single.pt',
    ModelName.YOLOv8ABOSEA: 'yolov8_abo_sea_single.pt',
}

current_folder = pathlib.Path(__file__).parent.resolve()
model_path_dict = {
    i: os.path.join(current_folder, model_names_to_file_names[i]) for i in model_names_to_file_names.keys()
}

model_download_links = {
    ModelName.YOLOv3ABO: 'https://drive.google.com/uc?id=13RQyPCxi-60SSM1ap2ynZXsNLoFtuwD7',
    ModelName.YOLOv3SEA: 'https://drive.google.com/uc?id=1L6RJPNoa72Jn0ZvkNGur4C1oUI-tzMuO',
    ModelName.YOLOv3ABOSEA: 'https://drive.google.com/uc?id=1RedHFJji0YLooQgkqT1xqRvKC9uP6oCi',
    ModelName.YOLOv5ABO: 'https://drive.google.com/uc?id=12GLNZ4fwhHINThwWPSXX6THemSYsi5we',
    ModelName.YOLOv5SEA: 'https://drive.google.com/uc?id=14N7gKT1E200J9_XoFe-1i2NFPUj1PGCs',
    ModelName.YOLOv5ABOSEA: 'https://drive.google.com/uc?id=1p_ZJzj80I-SBLwWZ1ZSD1zxSHFtMOn1q',
    ModelName.YOLOv8ABO: 'https://drive.google.com/uc?id=1RedHFJji0YLooQgkqT1xqRvKC9uP6oCi',
    ModelName.YOLOv8SEA: 'https://drive.google.com/uc?id=1RD0eP9bcpENITpnssM8Q0G8VWBb7vx9O',
    ModelName.YOLOv8ABOSEA: 'https://drive.google.com/uc?id=1OEvB9YlA8cvKgo_9tOQZqFGfSs1qentP'
}


def run_downloads():
    for i in model_path_dict.keys():
        if not os.path.exists(model_path_dict[i]):
            logging.info('{} at path {} does not exists! Starting download!'.format(
                i,
                model_path_dict[i]
            ))
            gdown.download(model_download_links[i], model_path_dict[i])
            logging.info('{} at path {} finished download!'.format(
                i,
                model_path_dict[i]
            ))
        else:
            logging.info('{} at path {} already exists!'.format(
                i,
                model_path_dict[i]
            ))
