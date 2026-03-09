import joblib
import os
import numpy as np
from meter import RunningMeter, BestMeter


def read_single_file(filename):
    with open(filename, "rb") as file:
        running_meter = joblib.load(file)

    max_val_f1score = np.max(np.asarray(running_meter.f1_score_weighted["val"]))

    return max_val_f1score


def read_folder(folder_name):
    # List all files in the directory specified by folder_name
    fscore_dict = {}
    for filename in os.listdir(folder_name):
        # Construct full file path
        file_path = os.path.join(folder_name, filename)

        # Check if it's a file and not a directory
        if os.path.isfile(file_path):
            # Call read_single_file for each file
            fscore_dict[file_path] = read_single_file(file_path)

    max_fscore_path = max(fscore_dict, key=lambda path: fscore_dict[path])
    max_fscore = fscore_dict[max_fscore_path]

    return max_fscore_path, max_fscore


if __name__ == "__main__":
    max_fscore_path, max_fscore = read_folder(
        "/coc/pcba1/mthukral3/gt/smarthome_discretization/lexicon-of-human-movements-smarthome_aruba_512emb/code/vq_cpc/saved_logs/Nov-20-2023"
    )

    print(max_fscore_path, max_fscore)
