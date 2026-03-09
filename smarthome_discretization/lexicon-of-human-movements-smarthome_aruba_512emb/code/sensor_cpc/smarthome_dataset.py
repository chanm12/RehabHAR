# custom smarthome dataset based on Sri's paper
import joblib

"""Cite: add sri's paper link"""


class CASASSlidingWindowDataset(Dataset):
    def __init__(self, x, y, window_size=20):
        self.x = x
        self.y = y
        self.window_size = window_size
        self.shape = self.__getshape__()
        self.size = self.__getsize__()

    def __getitem__(self, index):
        x = self.x[index : index + self.window_size]
        y = max(self.y[index + self.window_size])
        return x, y

    def __len__(self):
        return len(self.x) - self.window_size

    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)

    def __getsize__(self):
        return self.__len__()


def load_dataset(args):
    batch_size = 64
    dataset_dict = joblib.load(
        "/coc/pcba1/mthukral3/gt/"
        "lexicon-of-human-movements-main/Harish_code/data/casas_aruba_datasets.joblib"
    )

    train_dataset = dataset_dict["train_dataset"]
    val_dataset = dataset_dict["val_dataset"]
    test_dataset = dataset_dict["test_dataset"]

    train_dataloader, val_dataloader, test_dataloader = [
        torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True, num_workers=0
        )
        for dataset in [train_dataset, val_dataset, test_dataset]
    ]

    data_loaders = {"train": train_dataloader, "val": val_dataloader}
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

    return data_loaders, dataset_sizes


if __name__ == "__main__":
    load_dataset()
