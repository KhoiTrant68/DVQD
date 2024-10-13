import sys

sys.path.append("..")
from src.utils.util_modules import instantiate_from_config

# Example datasets configuration

data_dict = {
    "target": "src.data.databuilder.DataModuleFromConfig",
    "params": {
        "batch_size": 2,
        "num_workers": 4,
        "train_val": False,
        "train": {
            "target": "src.data.imagenet.ImageNetDataset",
            "params": {
                "split": "train",
                "data_dir": "/home/infordio-ai/khoi/HU/dataset/imagenet-mini/train",
                "size": 256,
                "random_crop": True,
            },
        },
        "val": {
            "target": "src.data.imagenet.ImageNetDataset",
            "params": {
                "split": "val",
                "data_dir": "/home/infordio-ai/khoi/HU/dataset/imagenet-mini/val",
                "size": 256,
                "random_crop": False,
            },
        },
    },
}


# DataModule setup
data_module = instantiate_from_config(data_dict)
# Get DataLoaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Iterate through DataLoader (Example)
for batch in train_loader:
    print(batch["input"].shape)
    break
