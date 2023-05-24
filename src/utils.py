import torch
import gc
import locale


def get_device_to_use():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("GPU is available: device set to GPU")
        device = torch.device("cuda:0")
    return device


def clean_env_after_one_run():
    gc.collect()
    torch.cuda.empty_cache()
    locale.getpreferredencoding = lambda: "UTF-8"
