import joblib
import numpy as np
import os
from datetime import date
from fairseq.data.dictionary import Dictionary
from ray import tune
from tqdm.auto import tqdm
from tslearn.piecewise import SymbolicAggregateApproximation

from dataset_locs import get_dataset_locs
from sliding_window import sliding_window
from utils import feature_save_name
from utils import set_all_seeds, update_args


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))

    # Just making it a vector if it was a 2D matrix
    data_y = np.reshape(data_y, (len(data_y),))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


def compute_sax_features(config, args=None):
    print("Inside compute features")

    # Adding the config params back into the arg parser
    args = update_args(config, args)
    args, _ = get_dataset_locs(args)
    print("Post update", args)

    # Setting seed after updating args in case the seed is updated
    set_all_seeds(args.random_seed)

    # Loading the data
    filename = os.path.join(args.root_dir, args.data_file)
    data_raw = joblib.load(filename)

    # Using the train data to compute the SAX transform
    mag = np.expand_dims(np.linalg.norm(data_raw["train"]["data"], axis=1), 1)
    train_data, train_labels = opp_sliding_window(
        mag, data_raw["train"]["labels"], args.window, args.overlap
    )

    # SAX transform
    n_paa_segments = args.window // args.span
    n_sax_symbols = args.num_sax_symbols
    sax = SymbolicAggregateApproximation(
        n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols
    )
    sax.fit(train_data)

    # Computing the features for all splits and storing them
    sax_data = {}
    for phase in ["train", "val", "test"]:
        sax_data[phase] = {}

        mag = np.expand_dims(np.linalg.norm(data_raw[phase]["data"], axis=1), 1)
        windowed_data, windowed_labels = opp_sliding_window(
            mag, data_raw[phase]["labels"], args.window, args.overlap
        )
        sax_data[phase]["data"] = sax.transform(windowed_data)
        sax_data[phase]["labels"] = windowed_labels

    # Next is to perform the actual indexing and storing in the format necessary
    dir_path = os.path.dirname(os.path.realpath(__file__))
    indexed_data_loc = os.path.join(
        dir_path,
        "indexed_data",
        date.today().strftime("%b-%d-%Y"),
        feature_save_name(args),
    )

    perform_indexing(sax_data, indexed_data_loc)

    tune.report(status=1)

    return


def perform_indexing(sax_data, indexed_data_loc):
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
    data = np.squeeze(sax_data["train"]["data"], axis=2)

    for i in tqdm(range(0, len(data))):
        data_str = [str(f) for f in data[i]]
        row = " ".join(data_str)
        _ = dictionary.encode_line(row)

    # Finalizing the dictionary creation
    dictionary.finalize()

    # Saving the dictionary into a dict file
    name = os.path.join(indexed_data_loc, "dict_mine.txt")
    dictionary.save(name)

    # Applying the dictionary to index the quantized data
    print("Applying the dictionary to all phases")
    for phase in ["train", "val", "test"]:
        data = np.squeeze(sax_data[phase]["data"], axis=2)
        indexed_data[phase]["labels"] = sax_data[phase]["labels"]

        for i in tqdm(range(0, len(data))):
            words = data[i]

            # Adding [0] and [2] for SOS and EOS tags. Getting the indices
            # for the rest of the codewords. <unk> is 3.
            indexed = [0] + [dictionary.index(str(w)) for w in words] + [2]

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
        joblib.dump(indexed_data, f)
    print("Saved the encoded data into a joblib file")

    return
