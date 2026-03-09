import copy
import numpy as np
import os
import torch
import torch.nn as nn
import wandb
from datetime import date
from ray import tune
from time import time
from torch import optim
from tqdm.auto import tqdm

from dataset import load_self_supervised_dataset
from dataset_locs import get_dataset_locs
from meter import RunningMeter, BestMeter
from model import TPNModel
from utils import (
    compute_best_metrics,
    compute_metrics,
    full_model_name,
    save_meter,
    update_args,
    set_all_seeds,
    save_model,
)


def learn_model(config, args=None):
    print("Inside pre-train")
    print(args)

    # Adding the config params back into the arg parser
    args, _ = get_dataset_locs(args)
    args = update_args(config, args)
    print("Post update", args)

    # Setting seed after updating args in case the seed is updated
    set_all_seeds(args.random_seed)

    # Initialize Wandb
    name = date.today().strftime("%b-%d-%Y") + "_" + full_model_name(args)
    wandb.init(
        config=args,
        project="quant_multi",
        name=name,
        settings=wandb.Settings(start_method="thread"),
    )

    # Loading the self-supervision processed dataset
    data_loaders, dataset_sizes = load_self_supervised_dataset(args)

    # Tracking meter
    running_meter = RunningMeter(args=args)
    best_meter = BestMeter()

    # Creating the model
    model = TPNModel(args).to(args.device)
    wandb.watch(model)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of trainable parameters is {}".format(num_parameters))

    # Optimizer settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Copying the randomly initialized weights for tracking best model
    best_model_wts = copy.deepcopy(model.state_dict())

    trigger_times = 0

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
            criterion,
            args,
            epoch,
            phase="val",
            dataset_size=dataset_sizes["val"],
            running_meter=running_meter,
        )

        # Saving the logs
        save_meter(args, running_meter)

        # Doing the early stopping check
        if epoch >= 2:
            if running_meter.loss["val"][-1] > best_meter.loss["val"]:
                trigger_times += 1
                print("Trigger times: {}".format(trigger_times))

                if trigger_times >= args.patience:
                    print(
                        "Early stopping the model at epoch: {}. The "
                        "validation loss has not improved for {}".format(
                            epoch, trigger_times
                        )
                    )
                    break
            else:
                trigger_times = 0
                print("Resetting the trigger counter for early stopping")

        # Updating the best weights
        if running_meter.loss["val"][-1] < best_meter.loss["val"]:
            print("Updating the best weights!")
            best_meter = compute_best_metrics(
                running_meter, best_meter, classifier=False
            )
            running_meter.update_best_meter(best_meter)
            print("Best weights at epoch: {}".format(running_meter.best_meter.epoch))

            best_model_wts = copy.deepcopy(model.state_dict())

            # Saving the logs
            save_meter(args, running_meter)

        # Printing the time taken
        time_elapsed = time() - since
        print(
            "Epoch {} completed in {:.0f}m {:.0f}s".format(
                epoch, time_elapsed // 60, time_elapsed % 60
            )
        )

        # For tuning with Ray
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(
            train_loss=running_meter.loss["train"][-1],
            val_loss=running_meter.loss["val"][-1],
        )

    # Printing the best metrics
    best_meter.display()

    # Updating the wandb summary metrics
    wandb.run.summary["best_train_loss"] = best_meter.loss["train"]
    wandb.run.summary["best_val_loss"] = best_meter.loss["val"]

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Saving the pretrained model
    save_model(args=args, model=model)

    return


def train(
    model, data_loader, criterion, optimizer, args, epoch, dataset_size, running_meter
):
    # Setting the model to training mode
    model.train()

    # To track the loss and other metrics
    running_loss = 0.0
    actual_labels = []
    pred_labels = []
    constitutent = {
        "multi": 0.0,
        "kmeans": 0.0,
        "multi_epoch": 0.0,
        "kmeans_epoch": 0.0,
    }

    # Iterating over the data
    for i, data in tqdm(enumerate(data_loader, 0)):
        loss = 0.0
        multi_loss = 0.0
        kmeans_loss = 0.0
        code_perplexity = []  # is averaged across all transformations

        optimizer.zero_grad()  # zero the gradients

        # Looping over all transformations
        for aug, d in data.items():
            inputs = d["data"].float().to(args.device)
            labels = d["label"].long().to(args.device)

            with torch.set_grad_enabled(True):
                outputs, quantizer_all = model(inputs, aug)
                _, preds = torch.max(outputs, 1)

                # Adding the kmeans loss if the quantization method is kmeans
                kmeans = 0
                if args.quantization_method == "kmeans":
                    kmeans = quantizer_all["kmeans_loss"]
                kmeans_loss += kmeans

                # Multi-task self-supervision loss
                multi = criterion(outputs, labels)
                multi_loss += multi

                # Total loss (i.e., if we are using kmeans quant, we have to
                # add it)
                loss += multi + kmeans

                # Appending predictions and labels
                actual_labels.extend(labels.cpu().data.numpy())
                pred_labels.extend(preds.cpu().data.numpy())

                # Appending the codebook perplexity
                code_perplexity.append(
                    quantizer_all["code_perplexity"].cpu().data.numpy()
                )

        # All the losses from the separate networks are backpropagated together
        loss.backward()
        optimizer.step()

        # Appending loss
        running_loss += loss.item() * inputs.size(0)
        constitutent["multi"] += multi_loss.item() * inputs.size(0)

        kmeans_temp = (
            kmeans_loss.item() * inputs.size(0)
            if args.quantization_method == "kmeans"
            else 0.0
        )
        constitutent["kmeans"] += kmeans_temp

        # Deeper tracking: code perplexity and losses per iteration
        phase = "train"
        wandb.log(
            {
                "Code perplexity/" + phase: np.mean(code_perplexity),
                "Iter loss/" + phase: loss.item() * inputs.size(0),
                "Iter kmeans loss/" + phase: kmeans_temp,
                "Iter multi loss/" + phase: multi_loss.item() * inputs.size(0),
            }
        )

    # Epoch Statistics
    loss = running_loss / dataset_size
    constitutent["multi_epoch"] = constitutent["multi"] / dataset_size
    constitutent["kmeans_epoch"] = constitutent["kmeans"] / dataset_size
    compute_metrics(
        actual_labels=actual_labels,
        pred_labels=pred_labels,
        phase="train",
        running_meter=running_meter,
        loss=loss,
        multi_loss=constitutent["multi_epoch"],
        kmeans_loss=constitutent["kmeans_epoch"],
        epoch=epoch,
        wandb=wandb,
    )

    return model, optimizer


def evaluate(
    model, data_loader, criterion, args, epoch, phase, dataset_size, running_meter
):
    # Setting the model to training mode
    model.eval()

    # To track the loss and other metrics
    running_loss = 0.0
    actual_labels = []
    pred_labels = []
    constitutent = {
        "multi": 0.0,
        "kmeans": 0.0,
        "multi_epoch": 0.0,
        "kmeans_epoch": 0.0,
    }

    # Iterating over the data
    for i, data in tqdm(enumerate(data_loader, 0)):
        loss = 0.0
        multi_loss = 0.0
        kmeans_loss = 0.0
        code_perplexity = []  # is averaged across all transformations

        # Looping over all transformations
        for aug, d in data.items():
            inputs = d["data"].float().to(args.device)
            labels = d["label"].long().to(args.device)

            with torch.set_grad_enabled(True):
                outputs, quantizer_all = model(inputs, aug)
                _, preds = torch.max(outputs, 1)

                # Adding the kmeans loss if the quantization method is kmeans
                kmeans = 0
                if args.quantization_method == "kmeans":
                    kmeans = quantizer_all["kmeans_loss"]
                kmeans_loss += kmeans

                # Multi-task self-supervision loss
                multi = criterion(outputs, labels)
                multi_loss += multi

                # Total loss (i.e., if we are using kmeans quant, we have to
                # add it)
                loss += multi + kmeans

                # Appending predictions and labels
                actual_labels.extend(labels.cpu().data.numpy())
                pred_labels.extend(preds.cpu().data.numpy())

                # Appending the codebook perplexity
                code_perplexity.append(
                    quantizer_all["code_perplexity"].cpu().data.numpy()
                )

        # Appending loss
        running_loss += loss.item() * inputs.size(0)
        constitutent["multi"] += multi_loss.item() * inputs.size(0)
        kmeans_temp = (
            kmeans_loss.item() * inputs.size(0)
            if args.quantization_method == "kmeans"
            else 0.0
        )
        constitutent["kmeans"] += kmeans_temp

        # Deeper tracking: code perplexity and losses per iteration
        wandb.log(
            {
                "Code perplexity/" + phase: np.mean(code_perplexity),
                "Iter loss/" + phase: loss.item() * inputs.size(0),
                "Iter kmeans loss/" + phase: kmeans_temp,
                "Iter multi loss/" + phase: multi_loss.item() * inputs.size(0),
            }
        )

    # Epoch Statistics
    loss = running_loss / dataset_size
    constitutent["multi_epoch"] = constitutent["multi"] / dataset_size
    constitutent["kmeans_epoch"] = constitutent["kmeans"] / dataset_size
    compute_metrics(
        actual_labels=actual_labels,
        pred_labels=pred_labels,
        phase=phase,
        running_meter=running_meter,
        loss=loss,
        multi_loss=constitutent["multi_epoch"],
        kmeans_loss=constitutent["kmeans_epoch"],
        epoch=epoch,
        wandb=wandb,
    )

    return
