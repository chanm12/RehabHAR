import numpy as np
import os
import torch
from datetime import date
from sklearn.metrics import accuracy_score, f1_score
import pickle


def feature_save_name(args, capture=False):
    dataset = "sax_{0.dataset}_w_{0.window}_o_{0.overlap}".format(args)

    # SAX params
    sax_params = "_span_{0.span}_num_sax_{0.num_sax_symbols}".format(args)

    # Target dataset fold
    fold = "_fold_0" if capture else "_fold_{0.fold}".format(args)

    name = dataset + sax_params + fold

    return name


def set_all_seeds(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    return


def update_args(config, args):
    if "dataset" in config:
        args.dataset = config["dataset"]
    if "fold" in config:
        args.fold = config["fold"]
    if "span" in config:
        args.span = config["span"]
    if "num_sax_symbols" in config:
        args.num_sax_symbols = config["num_sax_symbols"]
    if "classifier_lr" in config:
        args.classifier_lr = config["classifier_lr"]
    if "classifier_wd" in config:
        args.classifier_wd = config["classifier_wd"]
    if "exp_name" in config:
        args.exp_name = config["exp_name"]
    if "random_seed" in config:
        args.random_seed = config["random_seed"]
    if "date" in config:
        args.date = config["date"]

    print("Completed updating args")

    return args


def compute_metrics(
    actual_labels, pred_labels, phase, running_meter, loss, epoch, wandb
):
    acc = accuracy_score(actual_labels, pred_labels)
    f_score_weighted = f1_score(actual_labels, pred_labels, average="weighted")
    f_score_macro = f1_score(actual_labels, pred_labels, average="macro")

    running_meter.update(phase, loss, acc, f_score_macro, f_score_weighted)

    # Updating wandb
    wandb.log(
        {
            "Accuracy/" + phase: acc,
            "f_score_weighted/" + phase: f_score_weighted,
            "f_score_macro/" + phase: f_score_macro,
            "Epoch loss/" + phase: loss,
            "epoch": epoch,
        }
    )

    # printing the metrics
    print(
        "The epoch: {} | phase: {} | loss: {:.4f} | accuracy: {:.4f} | "
        "mean f1-score: {:.4f} | weighted f1-score: {:.4f}".format(
            epoch, phase, loss, acc, f_score_macro, f_score_weighted
        )
    )

    return


def compute_best_metrics(running_meter, best_meter, classifier=False):
    """
    To compute the best validation scores (f1 score) from the running meter
    object
    :param running_meter: running meter object with all values
    :param best_meter: updating the best meter based on current running meter
    :param cutoff: to compute the best val accuracy after x% of epochs are
    complete
    :return: best accuracy value
    """
    if classifier:
        loc = np.argmax(running_meter.f1_score["val"])
    else:
        min_loss = np.min(running_meter.loss["val"])  # Minimum loss
        loc = np.where(running_meter.loss["val"] == min_loss)[0][
            -1
        ]  # The latest epoch to give the lowest loss

    # Epoch where the best validation accuracy was obtained
    # Since we don't consider limit epochs, we add it back
    epoch = running_meter.epochs[loc]

    # Updating the best meter with values based on the epoch
    phases = ["train", "val", "test"]
    for phase in phases:
        best_meter.update(
            phase,
            running_meter.loss[phase][loc],
            running_meter.accuracy[phase][loc],
            running_meter.f1_score[phase][loc],
            running_meter.f1_score_weighted[phase][loc],
            epoch,
        )

    return best_meter


def save_meter(args, running_meter):
    save_name = feature_save_name(args) + "_log.pkl"

    # Creating logs by the dat now. To make stuff easier
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(dir_path, "saved_logs", date.today().strftime("%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)

    with open(
        os.path.join(
            dir_path, "saved_logs", date.today().strftime("%b-%d-%Y"), save_name
        ),
        "wb",
    ) as f:
        pickle.dump(running_meter, f, pickle.HIGHEST_PROTOCOL)

    return
