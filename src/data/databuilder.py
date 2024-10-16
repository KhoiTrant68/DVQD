from torch.utils.data import DataLoader

from src.utils.util_modules import instantiate_from_config


class DataModuleFromConfig:
    """Module to configure data loading from various dataset splits and options."""

    def __init__(
        self,
        batch_size,
        train=None,
        val=None,
        test=None,
        wrap=False,
        num_workers=None,
        train_val=False,
    ):
        """
        Initialize the DataModuleFromConfig.

        Args:
            batch_size (int): The number of samples per batch to load.
            train (dict, optional): Configuration for the training dataset.
            val (dict, optional): Configuration for the validation dataset.
            test (dict, optional): Configuration for the test dataset.
            wrap (bool, optional): Whether to wrap datasets with additional functionality.
            num_workers (int, optional): Number of subprocesses to use for data loading.
            train_val (bool, optional): Whether to combine train and validation datasets.
        """
        super().__init__()
        self.batch_size = batch_size
        self.train_val = train_val
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        if train is not None:
            self.dataset_configs["train"] = train
        if val is not None:
            self.dataset_configs["val"] = val
        if test is not None:
            self.dataset_configs["test"] = test
        self.wrap = wrap
        self.datasets = self.setup_datasets()

    def setup_datasets(self):
        """
        Set up datasets based on the provided configurations.

        Returns:
            dict: A dictionary containing instantiated datasets for each split.
        """
        datasets = {
            k: instantiate_from_config(self.dataset_configs[k])
            for k in self.dataset_configs
        }

        if self.train_val and "train" in datasets and "val" in datasets:
            datasets["train"] = datasets["train"] + datasets["val"]
        return datasets

    def get_dataloader(self, split):
        """
        Get a DataLoader for a specific dataset split.

        Args:
            split (str): The dataset split to load ('train', 'val', or 'test').

        Returns:
            DataLoader: A DataLoader instance for the specified dataset split.
        """
        dataset = self.datasets[split]
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=split == "train",
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("val")

    def test_dataloader(self):
        return self.get_dataloader("test")
