import numpy as np
import os
import pandas as pd
import pickle
from datetime import date
from fairseq.data.dictionary import Dictionary
from joblib import dump
from ray import tune
from time import time
from tqdm.auto import tqdm

from smarthome_dataset import load_classifier_dataset
from dataset_locs import get_dataset_locs
from model import QuantizedFeatureExtractor
from utils import model_save_name, update_args, set_all_seeds

from arguments import parse_args


def perform_quantization(args):
    """
    To analyze the quantization performed
    :param args: arguments
    :return: Nothing. It saves the data into a pickle file.
    """
    # Getting the trained model name
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args.saved_model_folder is not None:
        args.saved_model = os.path.join(
            dir_path,
            "models",
            args.saved_model_folder,
            model_save_name(args, capture=True) + ".pkl",
        )
    else:
        args.saved_model = None
    print("args",args)

    print("before data loading")
    # Data Loader
    data_loaders, dataset_sizes = load_classifier_dataset(args)

    # Creating the model
    model = QuantizedFeatureExtractor(args).to(args.device)

    if args.saved_model is not None:
        print("Loading learned weights")
        model.load_pretrained_weights(args)

    # Creating the folder for the quantized data and everything else necessary
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(
        dir_path,
        "indexed_data",
        date.today().strftime("%b-%d-%Y"),
        model_save_name(args, classifier=False),
    )
    os.makedirs(folder, exist_ok=True)
    quantized_data_loc = os.path.join(folder, "quantized_data.joblib")

    print("Performing the inference on the model")
    model.eval()  # Since we do only inference

    # Setting the number of encoded steps based on the downsampling performed
    # on the input
    # if args.input_downsampling == 2:
    #     num_steps = 19
    # elif args.input_downsampling == 4:
    #     num_steps = 23
    # elif args.input_downsampling == 1:
    #     num_steps = 97

    # as we dont have any conv num_steps same as window
    
    num_steps = args.window
    data = {
        "train": {"data": [], "labels": []},
        "val": {"data": [], "labels": []},
        "test": {"data": [], "labels": []},
    }
    since = time()
    for phase in ["train", "val", "test"]:
        encoded_data = np.zeros((dataset_sizes[phase], num_steps, 2))
        all_labels = np.zeros((dataset_sizes[phase],))
        count = 0

        for i, ( inputs, labels, attention_mask) in enumerate(data_loaders[phase], 0):
            inputs = inputs.long().to(args.device)
            attention_mask = attention_mask.int().to(args.device)
            labels = labels.long().to(args.device)

            # We take the output from the penultimate layer instead
            quantization_all = model(inputs, attention_mask)
            quantized_output = quantization_all["targets"]

            encoded_data[
                count : count + args.classifier_batch_size, ...
            ] = quantized_output.cpu().data.numpy()
            all_labels[
                count : count + args.classifier_batch_size
            ] = labels.cpu().data.numpy()
            count += args.classifier_batch_size

        # Saving the data into .npy files
        data[phase]["data"] = encoded_data
        data[phase]["labels"] = all_labels

    time_taken = time() - since
    print("The time taken to compute all features is {} seconds".format(time_taken))
    print(
        "The average time for computing a representation is {} seconds".format(
            time_taken
            / (
                data["train"]["data"].shape[0]
                + data["val"]["data"].shape[0]
                + data["test"]["data"].shape[0]
            )
        )
    )

    for phase in ["train", "val", "test"]:
        print(
            "The shape of {} data is {}, and train labels is {}".format(
                phase, data[phase]["data"].shape, data[phase]["labels"].shape
            )
        )

    with open(quantized_data_loc, "wb") as f:
        dump(data, f)
    print("Saved the encoded data into a joblib file")

    return data


def create_vocabulary(all_data, args):
    # If vocabulary is available, exit the function.
    indexed_data_loc = os.path.join(
        "indexed_data",
        date.today().strftime("%b-%d-%Y"),
        model_save_name(args, classifier=True) + "_vocab.joblib",
    )
    if os.path.exists(indexed_data_loc):
        print("Loading the prepared vocabulary")
        vocab = pd.read_pickle(indexed_data_loc)
        return vocab

    vocab = {"oov": 0}

    # Reshaping into a single column of lists
    train = all_data["train"]["data"]
    train = np.reshape(train, (train.shape[0] * train.shape[1], train.shape[2]))
    print("The shape of train dataset after reshaping is: {}".format(train.shape))

    # Passing through the column to compute the unique rows and to index them
    count = 1
    print("\nCreaing the vocabulary")
    for i in tqdm(range(len(train))):
        row = str(train[i])
        if row not in vocab:
            vocab[row] = count
            count += 1

    # Saving the vocab
    print("The size of the vocabulary is: {} codewords".format(count))

    # Saving the indexed data
    os.makedirs(
        os.path.join("indexed_data", date.today().strftime("%b-%d-%Y")), exist_ok=True
    )

    with open(indexed_data_loc, "wb") as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    print("Saved the vocab into a pickle file")

    return vocab


def compute_hyphenated_data(all_data, args):
    print("Hyphenating the data first")
    hyphenated_data = {
        "train": {"data": [], "labels": []},
        "val": {"data": [], "labels": []},
        "test": {"data": [], "labels": []},
    }

    # Indexing the data based on the vocabulary available
    for phase in ["train", "val", "test"]:
        # Copying over the labels
        hyphenated_data[phase]["labels"] = all_data[phase]["labels"]

        data = all_data[phase]["data"]
        print(
            "The phase is: {}. The shape of quantized data is: {}".format(
                phase, data.shape
            )
        )

        for frame in tqdm(range(len(data))):
            row = data[frame, :]
            row_string = " ".join("-".join(map(str, a.tolist())) for a in row)
            row_string = row_string.split(" ")
            hyphenated_data[phase]["data"].append(row_string)

    # Saving the hyphenated data into text files
    indexed_data_loc = save_hyphenated_data(hyphenated_data, args)

    return hyphenated_data, indexed_data_loc


def perform_indexing(hyphenated_data, indexed_data_loc):
    print("Indexing the hyphenated data")
    # Creating the dictionary object
    dictionary = Dictionary()

    # To store the indexed data for easy pickling
    indexed_data = {
        "train": {"data": [], "labels": []},
        "val": {"data": [], "labels": []},
        "test": {"data": [], "labels": []},
    }

    # Computing the dictionary on the train data
    print("\nComputing the dictionary on the train dataset")
    data = hyphenated_data["train"]["data"]

    for i in tqdm(range(0, len(data))):
        row = " ".join(data[i])
        _ = dictionary.encode_line(row)

    # Finalizing the dictionary creation
    dictionary.finalize()

    # Saving the dictionary into a dict file
    name = os.path.join(indexed_data_loc, "dict_mine.txt")
    dictionary.save(name)

    # Applying the dictionary to index the quantized data
    print("Applying the dictionary to all phases")
    for phase in ["train", "val", "test"]:
        data = hyphenated_data[phase]["data"]
        indexed_data[phase]["labels"] = hyphenated_data[phase]["labels"]

        for i in tqdm(range(0, len(data))):
            words = data[i]

            # Adding [0] and [2] for SOS and EOS tags. Getting the indices
            # for the rest of the codewords. <unk> is 3.
            indexed = [0] + [dictionary.index(w) for w in words] + [2]
            indexed_data[phase]["data"].append(indexed)

        indexed_data[phase]["data"] = np.array(indexed_data[phase]["data"])
        print(
            "The phase is: {}. The size is: {}".format(
                phase, indexed_data[phase]["data"].shape
            )
        )
    print("Indexing complete!")

    # Saving the indexed data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    indexed_data_loc = os.path.join(dir_path, indexed_data_loc, "data.joblib")

    with open(indexed_data_loc, "wb") as f:
        dump(indexed_data, f)
    print("Saved the encoded data into a joblib file")

    return


def save_hyphenated_data(data, args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    indexed_data_loc = os.path.join(
        dir_path,
        "indexed_data",
        date.today().strftime("%b-%d-%Y"),
        model_save_name(args, classifier=False),
    )

    os.makedirs(indexed_data_loc, exist_ok=True)

    for phase in ["train", "val", "test"]:
        df = pd.DataFrame(data[phase]["data"])
        df.to_csv(
            os.path.join(indexed_data_loc, phase + ".src"),
            index=None,
            header=None,
            sep=" ",
        )

    return indexed_data_loc


def generate_quantized_data(config, args=None):
    print("Starting the quantization process.")
    print(args)

    # Adding the config params back into the arg parser
    args.overlap = int(args.window // 2)
    args = update_args(config, args)
    args, _ = get_dataset_locs(args)
    print("Post update:\n", args)

    # Setting seed after updating args in case the seed is updated
    set_all_seeds(args.random_seed)

    # Obtaining the tuples closest to each data point
    tuples_from_cluster = perform_quantization(args)

    # Obtaining the hyphenated data
    hyphenated_data, indexed_data_loc = compute_hyphenated_data(
        tuples_from_cluster, args
    )

    # Perform indexing on the hyphenated data using the fairseq dictionary
    # creator
    perform_indexing(hyphenated_data, indexed_data_loc)

    # tune.report(done=1)

    return


if __name__ == "__main__":
    args = parse_args()
    print(args)

    generate_quantized_data(config={}, args=args)
