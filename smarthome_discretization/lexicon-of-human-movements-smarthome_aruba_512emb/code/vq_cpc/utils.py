import pickle
import subprocess
from datetime import date

import math
import numpy as np
import os
import random
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def model_save_name(args, classifier=False, capture=False, index=False):
    if capture:
        cpc = 'aruba_k_{0.num_steps_prediction}_neg_' \
              '{0.num_negatives}_enc_{0.input_downsampling}' \
              '_agg_{0.aggregator_type}_ws_{0.window}_bs_{0.batch_size}_over_{0.overlap}'.format(args)
    else:
        cpc = '{0.dataset}_k_{0.num_steps_prediction}_neg_' \
              '{0.num_negatives}_enc_{0.input_downsampling}' \
              '_agg_{0.aggregator_type}_ws_{0.window}_bs_{0.batch_size}_over_{0.overlap}'.format(args)

    # num layers from the conv aggregator (if we are using it)
    num_agg_layers = ""
    if args.aggregator_type == 'conv':
        num_agg_layers = "_{0.num_conv_agg_layers}".format(args)

    # Quantization
    quant = ""
    if args.vq_type == 'kmeans':
        quant += '_{0.vq_type}_{0.vq_gamma}_{0.groups}_{0.num_vars}' \
            .format(args)
    elif args.vq_type == 'gumbel':
        quant += '_{0.vq_type}_{0.gumbel_temperature}_{0.groups}_' \
                 '{0.num_vars}'.format(args)

    # CPC training settings
    training_settings = '_lr_{0.learning_rate}_wd_{0.weight_decay}_bs' \
                        '_{0.batch_size}'.format(args)

    # Classifier
    classification = ""
    if classifier:
        if args.saved_model is not None:
            classification += "_saved_model_True"

        classification += (
            "_cls_lr_{0.classifier_lr}_cls_wd"
            "_{0.classifier_wd}_{0.learning_schedule}_"
            "cls_bs_{0.classifier_batch_size}_"
            "{0.classifier_type}".format(args)
        )

    # Different random seed
    random_seed = "_rs_{0.random_seed}".format(args) if not index else \
        "_rs_42"

    # Different folds for testing
    fold = "_0" if capture else "_{0.fold}".format(args)

    # Experiment name
    exp_name = ""
    if classifier and args.exp_name != "":
        exp_name = "_" + args.exp_name

    # Pre-training Experiment name
    pre_exp_name = ""
    if not classifier and args.pre_exp_name != "":
        pre_exp_name = "_" + args.pre_exp_name

    caption_type = str(args.caption_type)
    name = cpc + num_agg_layers + quant + training_settings + classification + \
           random_seed + fold + pre_exp_name + exp_name + caption_type

    
    # name = args.saved_model.split("/")[-1].split(".")[0] + classification

    return name


def compute_best_metrics(running_meter, best_meter, classifier=False):
    """
    To compute the best validation loss from the running meter object
    :param running_meter: running meter object with all values
    :param best_meter: updating the best meter based on current running meter
    :return: best validation f1-score
    """
    if classifier:
        loc = np.argmax(running_meter.f1_score["val"])
    else:
        min_loss = np.min(running_meter.loss["val"])  # Minimum loss
        loc = np.where(running_meter.loss["val"] == min_loss)[0][
            -1
        ]  # The latest epoch to give the lowest loss

    # Epoch where the best validation loss was obtained
    epoch = running_meter.epochs[loc]

    # Updating the best meter with values based on the epoch
    phases = ["train", "val", "test"] if classifier else ["train", "val"]
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


def update_loss(phase, running_meter, loss, accuracy, epoch, wandb=None):
    running_meter.update(phase, loss, accuracy, 0, 0)

    # printing the metrics
    print(
        "The epoch: {} | phase: {} | loss: {:.4f} | accuracy: {:.4f} | mean "
        "f1-score: {:.4f} | weighted f1-score: {:.4f}".format(
            epoch, phase, loss, accuracy, 0, 0
        )
    )

    # Updating weights and biases
    wandb.log(
        {"Accuracy/" + phase: accuracy, "Epoch loss/" + phase: loss, "epoch": epoch}
    )

    return


def save_meter(args, running_meter, finetune=False):
    """
    Saving the logs
    :param args: arguments
    :param running_meter: running meter object to save
    :param mlp: if saving during the MLP training, then adds '_eval_log.pkl'
    to the end
    :return: nothing
    """
    name = model_save_name(args, classifier=finetune)
    save_name = name + "_finetune_log.pkl" if finetune else name + "_log.pkl"

    # Creating logs by the date now. To make stuff easier
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(dir_path, "saved_logs", date.today().strftime("%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, save_name), "wb") as f:
        pickle.dump(running_meter, f, pickle.HIGHEST_PROTOCOL)

    return


def save_model(model, args, epoch, classifier=False):
    """
    Saves the weights from the model
    :param model: model being trained
    :param args: arguments
    :param epoch: the epoch at which the model is being saved
    :param classifier: if we are training a classifier
    :return: nothing
    """
    name = model_save_name(args, classifier=classifier)

    # Creating logs by the date now. To make stuff easier
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(dir_path, "models", date.today().strftime("%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)

    model_name = os.path.join(folder, name + ".pkl")

    torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, model_name)

    return


def set_all_seeds(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.manual_seed(random_seed)

    return


def compute_classifier_metrics(
    actual_labels, pred_labels, phase, running_meter, loss, epoch, wandb=None
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
        "The epoch: {} | phase: {} | loss: {:.4f} | accuracy: {:.4f} | mean "
        "f1-score: {:.4f} | weighted f1-score: {:.4f}".format(
            epoch, phase, loss, acc, f_score_macro, f_score_weighted
        )
    )

    return running_meter


def update_args(config, args):
    if "dataset" in config:
        args.dataset = config["dataset"]
    if "num_steps_prediction" in config:
        args.num_steps_prediction = config["num_steps_prediction"]
    if "num_negatives" in config:
        args.num_negatives = config["num_negatives"]
    if "learning_schedule" in config:
        args.learning_schedule = config["learning_schedule"]
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
    if "batch_size" in config:
        args.batch_size = config["batch_size"]
    if "saved_model_folder" in config:
        args.saved_model_folder = config["saved_model_folder"]
    if "saved_model" in config:
        args.saved_model = config["saved_model"]
    if "data_file" in config:
        args.data_file = config["data_file"]
    if "root_dir" in config:
        args.root_dir = config["root_dir"]
    if "random_seed" in config:
        args.random_seed = config["random_seed"]
    if "fold" in config:
        args.fold = config["fold"]
    if "data_perc" in config:
        args.data_perc = config["data_perc"]
    if "aggregator_type" in config:
        args.aggregator_type = config["aggregator_type"]
    if "num_conv_agg_layers" in config:
        args.num_conv_agg_layers = config["num_conv_agg_layers"]
    if "input_downsampling" in config:
        args.input_downsampling = config["input_downsampling"]

        if args.input_downsampling == 4:
            args.conv_feature_layers = (
                "[(32, 4, 2), (64, 4, 2), (128, 1, 1)," " (256, 1, 1)]"
            )
        elif args.input_downsampling == 2:
            args.conv_feature_layers = (
                "[(32, 4, 2), (64, 1, 1), (128, 1, 1)," " (256, 1, 1)]"
            )
        elif args.input_downsampling == 1:
            args.conv_feature_layers = (
                "[(32, 4, 1), (64, 1, 1), (128, 1, 1)," " (256, 1, 1)]"
            )

    if "vq_type" in config:
        args.vq_type = config["vq_type"]
    if "vq_gamma" in config:
        args.vq_gamma = config["vq_gamma"]
    if "input_type" in config:
        args.input_type = config["input_type"]
    if "quant_date" in config:
        args.quant_date = config["quant_date"]
    if "gumbel_temperature" in config:
        args.gumbel_temperature = config["gumbel_temperature"]

    # Varying the number of variables and groups used in discretization
    if "num_vars" in config:
        args.num_vars = config["num_vars"]
    if "groups" in config:
        args.groups = config["groups"]

    # Type of classifier
    if "classifier_type" in config:
        args.classifier_type = config["classifier_type"]

    print("Completed updating args")
    return args


def get_git_revision_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


# Taken from: https://github.com/huggingface/transformers/blob/main/src
# /transformers/optimization.py
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the
    values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which
    it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to
            just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
