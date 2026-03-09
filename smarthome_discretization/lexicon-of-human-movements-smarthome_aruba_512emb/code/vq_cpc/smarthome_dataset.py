# custom smarthome dataset based on Sri's paper
import joblib
from torch.utils.data import Dataset
import torch
from torch.utils.data import Dataset, DataLoader
from sliding_window import sliding_window
import numpy as np
from transformers import AutoTokenizer, AutoModel
from scipy.stats import mode
import os
from utils import * 

"""Cite: add sri's paper link"""


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))

    # Just making it a vector if it was a 2D matrix
    data_y = np.reshape(data_y, (len(data_y),))
    #mode not last
    data_y = np.asarray([[max(i)] for i in sliding_window(data_y, ws, ss)])
  
    return data_x, data_y.reshape(len(data_y)).astype(np.uint8)


class CASASSlidingWindowDataset(Dataset):
    def __init__(self, x, y, window_size=20, overlap=10):
        self.window_size = window_size
        self.x, self.y = opp_sliding_window(x, y, window_size, overlap)
        print(self.x.shape, self.y.shape)

    def __getitem__(self, index):
        # no overlapp
        # start = index * self.window_size
        # print(index)

        # x = self.x[start:start+self.window_size]
        # y = max(self.y[index+self.window_size])

        x = self.x[index, :, :]

        x = torch.from_numpy(x).double()

        label = torch.from_numpy(np.asarray(self.y[index])).double()

        # print(x.shape, label)
        return x, label

    def __len__(self):
        return self.x.shape[0]


class SmartHomeDataset(Dataset):
    def __init__(self, x, y, window_size=20, overlap=10):
        self.window_size = window_size
        self.x, self.y = opp_sliding_window(x, y, window_size, overlap)
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        print(self.x.shape, self.y.shape)

    def __getitem__(self, index):
        # no overlapp
        # start = index * self.window_size
        # print(index)

        # x = self.x[start:start+self.window_size]
        # y = max(self.y[index+self.window_size])

        x = self.x[index, :, :]
        
        x= x.flatten().tolist()
       
        tokenized_sentences = self.tokenizer(x, padding="max_length", truncation=True, max_length=50,return_tensors='pt')
        text_data = torch.from_numpy(np.array(tokenized_sentences['input_ids']))
        attention_mask = torch.from_numpy(np.array(
            tokenized_sentences['attention_mask']))

        y= np.asarray(self.y[index])
        label = torch.from_numpy(y).double()
        

        return text_data, label, attention_mask

    def __len__(self):
        return self.x.shape[0]
    
class QuantizedClassifierDataset(Dataset):
    def __init__(self, loc, phase):
        self.all_data = joblib.load(os.path.join(loc, "data.joblib"))

        self.data = self.all_data[phase]["data"]
        self.labels = self.all_data[phase]["labels"]

        # Vocabulary size
        self.vocab_size = len(np.unique(self.all_data["train"]["data"])) + 4
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


def create_dataset(args):
    splits = {"train": 0.6, "val": 0.2, "test": 0.2}
    if args.caption_type=="numerical_values":
        filepath =  "/mnt/attached1/smarthome_datasets/aruba/all_data/preprocessed_aruba_without_time.joblib"
    elif args.caption_type=="numerical_values_as_text":
        filepath =  "/mnt/attached1/smarthome_datasets/aruba/all_data/preprocessed_aruba_without_time_with_text_values.joblib"    
        
    casas_dict = joblib.load(filepath)

    # x = casas_dict["X"][:10000,:]
    # y = casas_dict["y"][:10000]
    
    x = casas_dict["X"]
    y = casas_dict["y"]
    
    full_dataset = SmartHomeDataset(x, y, window_size=args.window, overlap=args.overlap)

    len_dataset = len(full_dataset)
    train_size = int(len_dataset * splits["train"])
    val_size = int(len_dataset * splits["val"])
    test_size = len_dataset - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    return train_dataset, val_dataset, test_dataset


def load_dataset(args):
    train_dataset, val_dataset, test_dataset = create_dataset(args)

    val_dataloader, test_dataloader = [
        torch.utils.data.DataLoader(
            dataset, 
            batch_size=128, 
            drop_last=True, 
            num_workers=0,
            shuffle=False,
        )
        for dataset in [ val_dataset, test_dataset]
    ]
    
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            drop_last=True,
            num_workers=0,
            shuffle=True,
        )

    

    data_loaders = {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
    }

    dataset_sizes = {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset),
    }

    return data_loaders, dataset_sizes


def load_classifier_dataset(args):
    train_dataset, val_dataset, test_dataset = create_dataset(args)

    val_dataloader, test_dataloader = [
        torch.utils.data.DataLoader(
            dataset,
            batch_size=args.classifier_batch_size,
            drop_last=True,
            num_workers=0,
        )
        for dataset in [val_dataset, test_dataset]
    ]

    train_dataloader =  torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.classifier_batch_size,
            drop_last=True,
            num_workers=0,
            shuffle=True
        )
    
    data_loaders = {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
    }

    dataset_sizes = {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset),
    }

    return data_loaders, dataset_sizes

def load_indexed_dataset(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    indexed_data_loc = os.path.join(
        dir_path,
        "indexed_data",
        args.quant_date,
        model_save_name(args, classifier=False, index=True),
    )
    print("The location is: {}".format(indexed_data_loc))
    

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


if __name__ == "__main__":
    load_dataset()
