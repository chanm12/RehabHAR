import numpy as np
import os
import torch
import torch.nn as nn
import wandb
from datetime import date

# Running experiments with Ray
from ray import tune
from time import time
from torch import optim
from tqdm.auto import tqdm

from dataset import load_classifier_dataset
from dataset_locs import get_dataset_locs
from meter import RunningMeter, BestMeter
from model import ClassificationModel
from utils import (
    compute_best_metrics,
    compute_metrics,
    full_model_name,
    save_meter,
    update_args,
    set_all_seeds,
)


# ------------------------------------------------------------------------------
def finetune(config, args=None):
    args.overlap = int(args.window // 2)
    print("Inside finetune")
    print(args)

    # Adding the config params back into the arg parser
    args = update_args(config, args)
    args, _ = get_dataset_locs(args)

    # Setting seed after updating args in case the seed is updated
    set_all_seeds(args.random_seed)

    # Getting the trained model name
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args.saved_model_folder is not None:
        args.saved_model = os.path.join(
            dir_path,
            "models",
            args.saved_model_folder,
            full_model_name(args, capture=True) + ".pkl",
        )
    else:
        args.saved_model = None

    print(args)

    # initialize Wandb
    name = (
        date.today().strftime("%b-%d-%Y") + "_" + full_model_name(args, classifier=True)
    )
    wandb.init(
        config=args,
        project="capture_24_classifier",
        name=name,
        settings=wandb.Settings(start_method="thread"),
    )

    # Loading the self-supervision processed dataset
    data_loaders, dataset_sizes = load_classifier_dataset(args)

    # Tracking meter
    running_meter = RunningMeter(args=args)
    best_meter = BestMeter()

    # Creating the model
    model = ClassificationModel(args=args)
    wandb.watch(model)  # Tracking the model with wandb

    if args.saved_model is not None:
        print("Loading learned weights")
        model.load_pretrained_weights(args)

    model = model.to(args.device)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of trainable parameters is {}".format(num_parameters))

    # Optimizer settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.classifier_lr, weight_decay=args.classifier_wd
    )
    print(optimizer)

    for epoch in range(0, args.num_epochs):
        since = time()

        # Training
        model, optimizer = train(
            model,
            data_loaders["train"],
            criterion,
            optimizer,
            args,
            epoch,
            dataset_sizes["train"],
            running_meter,
        )

        # Evaluating on the validation data
        evaluate(
            model,
            data_loaders["val"],
            args,
            criterion,
            epoch,
            phase="val",
            dataset_size=dataset_sizes["val"],
            running_meter=running_meter,
        )

        # Evaluating on the test data
        evaluate(
            model,
            data_loaders["test"],
            args,
            criterion,
            epoch,
            phase="test",
            dataset_size=dataset_sizes["test"],
            running_meter=running_meter,
        )

        # Saving the logs
        save_meter(args, running_meter, finetune=True)

        # Printing the time taken
        time_elapsed = time() - since
        print(
            "Epoch {} completed in {:.0f}m {:.0f}s".format(
                epoch, time_elapsed // 60, time_elapsed % 60
            )
        )

        tune.report(
            loss=running_meter.loss["val"][-1],
            val_f1_score=running_meter.f1_score["val"][-1],
            test_f1_score=running_meter.f1_score["test"][-1],
        )

    # Updating the best weights
    best_meter = compute_best_metrics(running_meter, best_meter, classifier=True)
    running_meter.update_best_meter(best_meter)
    save_meter(args, running_meter, finetune=True)
    print("Best weights at epoch: {}".format(running_meter.best_meter.epoch))
    best_meter.display()

    # Doing a sanity check that the best meter f1 score is the same as what
    # we get manually
    best_loc = np.argmax(running_meter.f1_score["val"])
    assert best_meter.epoch == best_loc
    assert best_meter.f1_score["val"] == running_meter.f1_score["val"][best_loc]
    assert best_meter.f1_score["test"] == running_meter.f1_score["test"][best_loc]

    # Updating the wandb summary metrics
    wandb.run.summary["best_test_f1"] = best_meter.f1_score["test"]
    wandb.run.summary["best_test_loss"] = best_meter.loss["test"]
    wandb.run.summary["best_test_acc"] = best_meter.accuracy["test"]

    return


def train(
    model, data_loader, criterion, optimizer, args, epoch, dataset_size, running_meter
):
    # Setting the model to training mode
    model.train()

    # If we wish to train only the last layer:
    if args.learning_schedule == "last_layer":
        model.freeze_encoder_layers()

    # To track the loss and other metrics
    running_loss = 0.0
    actual_labels = []
    pred_labels = []

    # Iterating over the data
    for i, (inputs, labels) in tqdm(enumerate(data_loader, 0)):
        inputs = inputs.float().to(args.device)
        labels = labels.long().to(args.device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        # Appending predictions and loss
        running_loss += loss.item() * inputs.size(0)
        actual_labels.extend(labels.cpu().data.numpy())
        pred_labels.extend(preds.cpu().data.numpy())

    # Statistics
    loss = running_loss / dataset_size
    compute_metrics(
        actual_labels=actual_labels,
        pred_labels=pred_labels,
        phase="train",
        running_meter=running_meter,
        loss=loss,
        epoch=epoch,
        wandb=wandb,
    )

    return model, optimizer


def evaluate(
    model, data_loader, args, criterion, epoch, phase, dataset_size, running_meter
):
    model.eval()

    # To track the loss and other metrics
    running_loss = 0.0
    actual_labels = []
    pred_labels = []

    # Iterating over the data
    for i, (inputs, labels) in tqdm(enumerate(data_loader, 0)):
        inputs = inputs.float().to(args.device)
        labels = labels.long().to(args.device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # Appending predictions and loss
        running_loss += loss.item() * inputs.size(0)
        actual_labels.extend(labels.cpu().data.numpy())
        pred_labels.extend(preds.cpu().data.numpy())

    # Statistics
    loss = running_loss / dataset_size
    compute_metrics(
        actual_labels=actual_labels,
        pred_labels=pred_labels,
        phase=phase,
        running_meter=running_meter,
        loss=loss,
        epoch=epoch,
        wandb=wandb,
    )

    return
