import numpy as np
import pickle
from datetime import date

import numpy as np
import os
import torch
from sklearn.metrics import accuracy_score, f1_score


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
    phases = ["train", "val", "test"] if classifier else ["train", "val"]
    for phase in phases:
        best_meter.update(
            phase,
            running_meter.loss[phase][loc],
            running_meter.kmeans_loss[phase][loc],
            running_meter.multi_loss[phase][loc],
            running_meter.accuracy[phase][loc],
            running_meter.f1_score[phase][loc],
            running_meter.f1_score_weighted[phase][loc],
            epoch,
        )

    return best_meter


def compute_metrics(
    actual_labels,
    pred_labels,
    phase,
    running_meter,
    loss,
    kmeans_loss,
    multi_loss,
    epoch,
    wandb,
):
    acc = accuracy_score(actual_labels, pred_labels)
    f_score_weighted = f1_score(actual_labels, pred_labels, average="weighted")
    f_score_macro = f1_score(actual_labels, pred_labels, average="macro")

    running_meter.update(
        phase, loss, kmeans_loss, multi_loss, acc, f_score_macro, f_score_weighted
    )

    # Updating wandb
    wandb.log(
        {
            "Accuracy/" + phase: acc,
            "f_score_weighted/" + phase: f_score_weighted,
            "f_score_macro/" + phase: f_score_macro,
            "Epoch loss/" + phase: loss,
            "Epoch kmeans loss/" + phase: kmeans_loss,
            "Epoch multi loss/" + phase: multi_loss,
            "epoch": epoch,
        }
    )

    # printing the metrics
    print(
        "The epoch: {} | phase: {} | loss: {:.4f} | kmeans loss: {:.4f} | "
        "multi loss: {:.4f} | accuracy: {:.4f} | mean f1-score: {:.4f} | "
        "weighted f1-score: {:.4f}".format(
            epoch,
            phase,
            loss,
            kmeans_loss,
            multi_loss,
            acc,
            f_score_macro,
            f_score_weighted,
        )
    )

    return


def full_model_name(args, classifier=False, capture=False):
    # First part is about the encoder
    encoder = (
        "multitask_{0.dataset}".format(args)
        if not capture
        else "multitask_capture_24".format(args)
    )

    # CPC training settings
    training_settings = (
        "_{0.quantization_method}_lr_{0.learning_rate}"
        "_wd_{0.weight_decay}_bs_{0.batch_size}".format(args)
    )

    # Classifier settings
    classification = ""
    if classifier:
        if args.saved_model is not None:  # i.e., we are using learned weights
            classification += "_saved_model_True"

        classification += (
            "_cls_lr_{0.classifier_lr}_cls_wd"
            "_{0.classifier_wd}_{0.learning_schedule}_"
            "cls_bs_{0.classifier_batch_size}".format(args)
        )

    # Few shot learning
    few_shot = ""
    if args.few_shot != 0:
        few_shot = "_few_shot_{0.few_shot}".format(args)

    # Different random seed
    random_seed = "_rs_{0.random_seed}".format(args)

    # Target dataset fold
    fold = "_multiple_runs_0" if capture else "_multiple_runs_{0.fold}".format(args)

    # Experiment name
    exp_name = ""
    if classifier and args.exp_name != "":
        exp_name = "_" + args.exp_name

    # Pre-training Experiment name
    pre_exp_name = ""
    if not classifier and args.pre_exp_name != "":
        pre_exp_name = "_" + args.pre_exp_name

    name = (
        encoder
        + training_settings
        + classification
        + few_shot
        + random_seed
        + fold
        + pre_exp_name
        + exp_name
    )

    return name


def set_softmax_layer_trainable(model):
    """
    To set only the softmax to be trainable
    :param model: classifier model
    :return: returning the same model but with only the last layer being
    trainable
    """
    print("Setting only the softmax layer to be trainable")
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        "Before setting, the number of trainable parameters is {}".format(
            num_parameters
        )
    )

    # First setting the model to eval
    model.eval()

    # Then setting the requires_grad to False
    for param in model.parameters():
        param.requires_grad = False

    # Setting the classifier layer to training mode
    model.classifier.train()

    # Setting the parameters in the softmax layer to tbe trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        "After setting, the number of trainable parameters is {}".format(num_parameters)
    )

    return model


def save_meter(args, running_meter, finetune=False):
    if finetune:
        save_name = full_model_name(args, classifier=True) + "_finetune_log.pkl"
    else:
        save_name = full_model_name(args) + "_log.pkl"

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


def save_model(args, model):
    name = full_model_name(args)

    # Creating logs by the dat now. To make stuff easier
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(dir_path, "models", date.today().strftime("%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)

    model_name = os.path.join(folder, name + ".pkl")

    torch.save(model.state_dict(), model_name)

    return


def update_args(config, args):
    if "dataset" in config:
        args.dataset = config["dataset"]
    if "learning_rate" in config:
        args.learning_rate = config["learning_rate"]
    if "weight_decay" in config:
        args.weight_decay = config["weight_decay"]
    if "classifier_lr" in config:
        args.classifier_lr = config["classifier_lr"]
    if "classifier_wd" in config:
        args.classifier_wd = config["classifier_wd"]
    if "exp_name" in config:
        args.exp_name = config["exp_name"]
    if "pre_exp_name" in config:
        args.pre_exp_name = config["pre_exp_name"]
    if "saved_model_folder" in config:
        args.saved_model_folder = config["saved_model_folder"]
    if "data_file" in config:
        args.data_file = config["data_file"]
    if "root_dir" in config:
        args.root_dir = config["root_dir"]
    if "random_seed" in config:
        args.random_seed = config["random_seed"]
    if "fold" in config:
        args.fold = config["fold"]
    if "quantization_method" in config:
        args.quantization_method = config["quantization_method"]
    if "data_perc" in config:
        args.data_perc = config["data_perc"]

    print("Completed updating args")

    return args


def set_all_seeds(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    return
