from time import time

import joblib
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sliding_window import sliding_window
from dataset_locs import get_dataset_locs
from utils import model_save_name
from sklearn.preprocessing import StandardScaler, LabelEncoder

class HARDataset(Dataset):
    def __init__(self, args, phase):
        print('Loading the normalized data: ', args.root_dir,
              args.data_file)
        phase_file = phase + '.joblib'
        self.filename = os.path.join(args.root_dir, phase_file)
        print(self.filename)

        # If the prepared dataset doesn't exist, give a message and exit
        if not os.path.isfile(self.filename):
            print('The data is not available. '
                  'Ensure that the data is present in the directory.')
            exit(0)

        # Loading the data from the .mat file
        self.data_raw = self.load_dataset(self.filename)
        assert args.input_size == self.data_raw['data'].shape[1]

        self.data, self.labels = self.data_raw['data'], self.data_raw['labels']

        print('The dataset is: {}. The phase is: {}. The size of the dataset '
              'is: {}'.format(args.dataset, phase, self.data.shape))

        self.window_size = args.window

        # Utilizing a fraction of the available data for pre-training
        data_perc = args.data_perc if phase == 'train' else 100.0
        num_total_windows = len(self.data) // self.window_size
        # print('Total windows possible: {}'.format(num_total_windows))
        num_windows = int((data_perc / 100.0) * num_total_windows)
        # print('The number of possible windows: {}'.format(num_windows))
        all_possible_indices = np.arange(num_total_windows) * 100
        # print('The possible indices: {}'.format(all_possible_indices))

        if phase == 'train':
            all_possible_indices = np.random.permutation(all_possible_indices)
        # print('After : {}'.format(all_possible_indices))
        self.subset_indices = np.sort(all_possible_indices[:num_windows])
        print('Num windows after subsetting: {}'
              .format(len(self.subset_indices)))
        self.num_windows = num_windows

    def load_dataset(self, filename):
        since = time()
        data_raw = joblib.load(filename)

        time_elapsed = time() - since
        print('Data loading completed in {:.0f}m {:.0f}s'
              .format(time_elapsed // 60, time_elapsed % 60))

        return data_raw

    def __len__(self):
        return self.num_windows

    def __getitem__(self, index):
        start = self.subset_indices[index]
        end = start + self.window_size

        data = self.data[start:end, :]
        data = torch.from_numpy(np.array(data))

        label = np.asarray(self.labels[index * self.window_size:
                                       (index + 1) * self.window_size])[-1]
        label = torch.from_numpy(np.array(label))

        return data, label


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))

    # Just making it a vector if it was a 2D matrix
    data_y = np.reshape(data_y, (len(data_y),))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)). \
        astype(np.uint8)


# Defining the data loader for the implementation
class ClassifierDataset(Dataset):
    def __init__(self, args, phase):
        print('Loading the normalized data: ', args.root_dir, args.data_file)
        self.filename = os.path.join(args.root_dir, args.data_file)
        print(self.filename)

        # If the prepared dataset doesn't exist, give a message and exit
        if not os.path.isfile(self.filename):
            print('The data is not available. '
                  'Ensure that the data is present in the directory.')
            exit(0)

        # Loading the data from the .mat file
        self.data_raw = self.load_dataset(self.filename)
        assert args.input_size == self.data_raw[phase]['data'].shape[1]

        # Applying the available capture 24 norms
        if phase=='train':
            scaler = StandardScaler()
            scaler.fit(self.data_raw[phase]['data'])
            joblib.dump(scaler, './pamap_scaler.joblib')
        else:
            scaler = joblib.load('./pamap_scaler.joblib')
            
        #_, scaler_loc = get_dataset_locs(args)
        #scaler = joblib.load(scaler_loc)
        print('Before using capture 24 norms: {}'
              .format(np.mean(self.data_raw[phase]['data'], axis=0)))
        self.data_raw[phase]['data'] = scaler. \
            transform(self.data_raw[phase]['data'])
        print('After using capture 24 norms: {}'
              .format(np.mean(self.data_raw[phase]['data'], axis=0)))

        # Obtaining the segmented data
        self.data, self.labels = \
            opp_sliding_window(self.data_raw[phase]['data'],
                               self.data_raw[phase]['labels'],
                               args.window, args.overlap)
            
        

    def load_dataset(self, filename):
        since = time()
        data_raw = joblib.load(filename)

        time_elapsed = time() - since
        print('Data loading completed in {:.0f}m {:.0f}s'
              .format(time_elapsed // 60, time_elapsed % 60))

        return data_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index, :, :]

        data = torch.from_numpy(data).double()

        label = torch.from_numpy(np.asarray(self.labels[index])).double()
        return data, label


class QuantizedClassifierDataset(Dataset):
    def __init__(self, loc, phase):
        self.all_data = joblib.load(os.path.join(loc, 'data.joblib'))

        self.data = self.all_data[phase]['data']
        self.labels = self.all_data[phase]['labels']

        # Vocabulary size
        self.vocab_size = len(np.unique(self.all_data['train']['data'])) + 4
        print('The size of the vocab is: {}'.format(self.vocab_size))

        not_present = len(list(set(np.ravel(self.all_data[phase]['data'])) -
                               set(np.ravel(self.all_data['train']['data']))))
        print('The number of codewords present in {} but not present in train '
              'are: {}'.format(phase, not_present))

        # Just printing
        print('The phase: {} | data size: {} | labels: {}'
              .format(phase, self.data.shape, self.labels.shape))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index, :]
        data = torch.from_numpy(data).long()

        label = self.labels[index]
        label = torch.from_numpy(np.asarray(label)).long()

        return data, label


def load_dataset(args):
    datasets = {x: HARDataset(args=args, phase=x) for x in
                ['train', 'val']}
    data_loaders = {x: DataLoader(datasets[x], batch_size=args.batch_size if
    x == 'train' else 256, shuffle=True if x == 'train' else False,
                                  num_workers=4, pin_memory=True) for x in
                    ['train', 'val']}

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    return data_loaders, dataset_sizes


def load_classifier_dataset(args):
    datasets = {x: ClassifierDataset(args=args, phase=x) for x in
                ['train', 'val', 'test']}
 
    processed = {'train': {'data': datasets['train'].data,
                           'labels': datasets['train'].labels},
                 'val': {'data': datasets['val'].data,
                         'labels': datasets['val'].labels},
                 'test': {'data': datasets['test'].data,
                          'labels': datasets['test'].labels},
                 }
 
    joblib.dump(processed, '/coc/pcba1/mthukral3/datasets/processed_data/discretization/segmented_pre_processed/pamap2/fold_'+str(args.fold)+'.joblib')
    
    data_loaders = {x: DataLoader(datasets[x],
                                  batch_size=args.classifier_batch_size if
    x == 'train' else 256, shuffle=False,
                                  num_workers=2, pin_memory=True) for x in
                    ['train', 'val', 'test']}

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}

    return data_loaders, dataset_sizes


def load_indexed_dataset(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    indexed_data_loc = os.path.join(dir_path,
                                    'indexed_data',
                                    args.quant_date,
                                    model_save_name(args, classifier=False,
                                                    index=True))
    print('The location is: {}'.format(indexed_data_loc))

    datasets = {x: QuantizedClassifierDataset(loc=indexed_data_loc, phase=x)
                    for x in ['train', 'val', 'test']}
    data_loaders = \
        {x: DataLoader(datasets[x],
                       batch_size=args.classifier_batch_size,
                       shuffle=True if x == 'train' else False,
                       num_workers=2,
                       pin_memory=True)
         for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}

    return data_loaders, dataset_sizes
