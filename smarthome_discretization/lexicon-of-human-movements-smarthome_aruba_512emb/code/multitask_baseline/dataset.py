from time import time

import joblib
import numpy as np
import os
import torch
from augmentation import (
    Jitter,
    Scaling,
    HorizontalFlipping,
    ChannelShuffling,
    Negation,
    Rotation,
    TimeWarping,
    Permutation,
)
from torch.utils.data import Dataset, DataLoader

from sliding_window import sliding_window
from dataset_locs import get_dataset_locs


class SelfSupervisedHARDataset(Dataset):
    """
    Defining the data loader for the multi-task baseline.
    """

    def __init__(self, args, phase):
        print("Loading the normalized data: ", args.root_dir, args.data_file)
        phase_file = phase + ".joblib"
        self.filename = os.path.join(args.root_dir, phase_file)

        # If the prepared dataset doesn't exist, give a message and exit
        if not os.path.isfile(self.filename):
            print("The preprocessed data is not available: {}".format(self.filename))
            exit(0)

        # Loading the data from the .mat file
        self.data_raw = self.load_dataset(self.filename)
        assert args.input_size == self.data_raw["data"].shape[1]

        # Data for segmentation
        self.data, self.labels = self.data_raw["data"], self.data_raw["labels"]
        self.window_size = args.window

        # Utilizing a fraction of the available data for pre-training
        data_perc = args.data_perc if phase == "train" else 100.0
        num_total_windows = len(self.data) // self.window_size
        # print('Total windows possible: {}'.format(num_total_windows))
        num_windows = int((data_perc / 100.0) * num_total_windows)
        # print('The number of possible windows: {}'.format(num_windows))
        all_possible_indices = np.arange(num_total_windows) * 100
        # print('The possible indices: {}'.format(all_possible_indices))

        if phase == "train":
            all_possible_indices = np.random.permutation(all_possible_indices)
        # print('After : {}'.format(all_possible_indices))
        self.subset_indices = np.sort(all_possible_indices[:num_windows])
        print("Num windows after subsetting: {}".format(len(self.subset_indices)))
        self.num_windows = num_windows

    def load_dataset(self, filename):
        since = time()
        data_raw = joblib.load(filename)

        time_elapsed = time() - since
        print(
            "Data loading completed in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

        return data_raw

    def __len__(self):
        return self.num_windows

    def __getitem__(self, index):
        d = self.data[index * self.window_size : (index + 1) * self.window_size, :]
        signal_transforms = {
            "noised": Jitter(),
            "scaled": Scaling(),
            "rotated": Rotation(),
            "negated": Negation(),
            "horizontally-flipped": HorizontalFlipping(),
            "permuted": Permutation(),
            "time-warped": TimeWarping(),
            "channel-shuffled": ChannelShuffling(),
        }

        data = {}
        for k, v in signal_transforms.items():
            datum, label = v(d)
            datum = np.expand_dims(datum, 0)
            datum = torch.from_numpy(datum).double()

            label = torch.from_numpy(np.asarray(label)).double()
            data[k] = {"data": datum, "label": label}

        return data


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))

    # Just making it a vector if it was a 2D matrix
    data_y = np.reshape(data_y, (len(data_y),))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


# Defining the data loader for the implementation
class ClassifierDataset(Dataset):
    def __init__(self, args, phase):
        print("Loading the normalized data: ", args.root_dir, args.data_file)
        self.filename = os.path.join(args.root_dir, args.data_file)
        print(self.filename)

        # If the prepared dataset doesn't exist, give a message and exit
        if not os.path.isfile(self.filename):
            print(
                "The data is not available. "
                "Ensure that the data is present in the directory."
            )
            exit(0)

        # Loading the data from the .mat file
        self.data_raw = self.load_dataset(self.filename)
        assert args.input_size == self.data_raw[phase]["data"].shape[1]

        # Applying the available capture 24 norms
        _, scaler_loc = get_dataset_locs(args)
        scaler = joblib.load(scaler_loc)
        print(
            "Before using capture 24 norms: {}".format(
                np.mean(self.data_raw[phase]["data"], axis=0)
            )
        )
        self.data_raw[phase]["data"] = scaler.transform(self.data_raw[phase]["data"])
        print(
            "After using capture 24 norms: {}".format(
                np.mean(self.data_raw[phase]["data"], axis=0)
            )
        )

        # Obtaining the segmented data
        self.data, self.labels = opp_sliding_window(
            self.data_raw[phase]["data"],
            self.data_raw[phase]["labels"],
            args.window,
            args.overlap,
        )

        print(
            "The dataset is: {}. The phase is: {}. The size of the dataset "
            "is: {}".format(args.dataset, phase, self.data.shape)
        )

    def load_dataset(self, filename):
        since = time()
        data_raw = joblib.load(filename)

        time_elapsed = time() - since
        print(
            "Data loading completed in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

        return data_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index, :, :]

        data = np.expand_dims(data, 0)
        data = torch.from_numpy(data).double()

        label = torch.from_numpy(np.asarray(self.labels[index])).double()
        return data, label


def load_classifier_dataset(args):
    datasets = {
        x: ClassifierDataset(args=args, phase=x) for x in ["train", "val", "test"]
    }
    data_loaders = {
        x: DataLoader(
            datasets[x],
            batch_size=args.batch_size if x == "train" else 256,
            shuffle=True if x == "train" else False,
            num_workers=1,
            pin_memory=True,
        )
        for x in ["train", "val", "test"]
    }

    dataset_sizes = {x: len(datasets[x]) for x in ["train", "val", "test"]}

    return data_loaders, dataset_sizes


def load_self_supervised_dataset(args):
    datasets = {
        x: SelfSupervisedHARDataset(args=args, phase=x) for x in ["train", "val"]
    }
    data_loaders = {
        x: DataLoader(
            datasets[x],
            batch_size=args.classifier_batch_size,
            shuffle=True if x == "train" else False,
            num_workers=2,
            pin_memory=True,
        )
        for x in ["train", "val"]
    }

    dataset_sizes = {x: len(datasets[x]) for x in ["train", "val"]}

    return data_loaders, dataset_sizes
