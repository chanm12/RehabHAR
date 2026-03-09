import joblib
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

from sliding_window import sliding_window
from utils import feature_save_name


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))

    # Just making it a vector if it was a 2D matrix
    data_y = np.reshape(data_y, (len(data_y),))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


class QuantizedClassifierDataset(Dataset):
    def __init__(self, loc, phase):
        self.all_data = joblib.load(os.path.join(loc, "data.joblib"))

        self.data = self.all_data[phase]["data"]
        self.labels = self.all_data[phase]["labels"]

        # Vocabulary size. The plus 2 is added for the UNK and padding tokens.
        self.vocab_size = len(np.unique(self.all_data["train"]["data"])) + 2
        print("The size of the vocab is: {}".format(self.vocab_size))

        not_present = len(
            list(
                set(np.ravel(self.all_data[phase]["data"]))
                - set(np.ravel(self.all_data["train"]["data"]))
            )
        )
        print(
            "The number of codewords present in {} but not present in train "
            "are: {}".format(phase, not_present)
        )

        # Just printing
        print(
            "The phase: {} | data size: {} | labels: {}".format(
                phase, self.data.shape, self.labels.shape
            )
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index, :]
        data = torch.from_numpy(data).long()

        label = self.labels[index]
        label = torch.from_numpy(np.asarray(label)).long()

        return data, label


def load_indexed_dataset(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    indexed_data_loc = os.path.join(
        dir_path, "indexed_data", args.date, feature_save_name(args)
    )

    datasets = {
        x: QuantizedClassifierDataset(loc=indexed_data_loc, phase=x)
        for x in ["train", "val", "test"]
    }
    data_loaders = {
        x: DataLoader(
            datasets[x],
            batch_size=args.classifier_batch_size,
            shuffle=True if x == "train" else False,
            num_workers=2,
            pin_memory=True,
        )
        for x in ["train", "val", "test"]
    }
    dataset_sizes = {x: len(datasets[x]) for x in ["train", "val", "test"]}

    return data_loaders, dataset_sizes
