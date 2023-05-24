from ultralytics import YOLO
from . import ModelName, model_path_dict, run_downloads, model_download_links
import os


class ModelFactory:
    @staticmethod
    def from_model_name(model_name: str) -> YOLO:
        mn = ModelName.from_str(model_name)
        if not os.path.exists(model_download_links[mn]):
            run_downloads()
        return YOLO(model=model_path_dict[mn])
