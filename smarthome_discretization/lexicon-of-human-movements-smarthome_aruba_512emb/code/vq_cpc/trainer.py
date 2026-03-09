from datetime import date
from time import time

import copy
import numpy as np
import torch
import torch.nn as nn
import wandb
from ray import tune
from sklearn.metrics import accuracy_score
from torch import optim
from tqdm.auto import tqdm

# from dataset import load_dataset
from smarthome_dataset import load_dataset
from dataset_locs import get_dataset_locs
from meter import RunningMeter, BestMeter
from model import CPC
from utils import (
    compute_best_metrics,
    update_loss,
    save_meter,
    model_save_name,
    set_all_seeds,
    save_model,
    update_args,
    get_cosine_schedule_with_warmup,
)
from arguments import parse_args


def learn_model(config):
    args = parse_args()
    print("Inside pre-train")
    print(args)

    # Getting the dataset locations based on args
    args, _ = get_dataset_locs(args)
    args = update_args(config, args)
    print("Post update", args)

    # Setting seed after updating args in case the seed is updated
    set_all_seeds(args.random_seed)

    # initialize Wandb
    name = date.today().strftime("%b-%d-%Y") + "_" + model_save_name(args)
    wandb.init(
        config=args,
        project="lexicon_smarthome_textdata",
        name=name,
        settings=wandb.Settings(start_method="fork"),
        mode='disabled'
       
    )

    # Loading the self-supervision processed dataset
    data_loaders, dataset_sizes = load_dataset(args)

    # Tracking meter
    running_meter = RunningMeter(args=args)
    best_meter = BestMeter()

    # Creating the model
    model = CPC(args).to(args.device)
    wandb.watch(model)
    best_model_wts = copy.deepcopy(model.state_dict())

    # Optimizer and criterion settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # The cosine scheduler for pre-training to see if it helps.
    # Currently, for 8% of windows, we have 3232 batches, and then the 0.1
    # for using the first 8% of updates only for warming up.
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_epochs
        * int(dataset_sizes["train"] // args.batch_size)
        * 0.08,
        num_training_steps=args.num_epochs
        * int(dataset_sizes["train"] // args.batch_size),
    )
    print(
        "Warmup updates: {}, total updates: {}".format(
            args.num_epochs * int(dataset_sizes["train"] // args.batch_size) * 0.08,
            args.num_epochs * int(dataset_sizes["train"] // args.batch_size),
        )
    )

    trigger_times = 0

    for epoch in range(0, args.num_epochs):
        since = time()

        # Training
        model, optimizer, scheduler = train(
            model,
            data_loaders["train"],
            criterion,
            optimizer,
            args,
            epoch,
            dataset_sizes["train"],
            running_meter,
            scheduler,
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
        if epoch >= 20:
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
            print(
                "Updating the best val loss at epoch: {}, since {} < "
                "{}".format(
                    epoch, running_meter.loss["val"][-1], best_meter.loss["val"]
                )
            )
            best_meter = compute_best_metrics(running_meter, best_meter)
            running_meter.update_best_meter(best_meter)

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
        # tune.report(
        #     train_loss=running_meter.loss["train"][-1],
        #     val_loss=running_meter.loss["val"][-1],
        #     train_acc=running_meter.accuracy["train"][-1],
        #     val_acc=running_meter.accuracy["val"][-1]
        # )

    # Printing the best metrics
    best_meter.display()

    # Updating the wandb summary metrics
    wandb.run.summary["best_train_loss"] = best_meter.loss["train"]
    wandb.run.summary["best_val_loss"] = best_meter.loss["val"]

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Saving the best performing model
    print("Updating the best trained model at epoch: {}!".format(epoch))
    save_model(model, args, epoch=epoch)

    return


def train(
    model,
    data_loader,
    criterion,
    optimizer,
    args,
    epoch,
    dataset_size,
    running_meter,
    scheduler=None,
):
    # Setting the model to training mode
    model.train()

    # To track the loss and other metrics
    running_loss = 0.0
    iter_acc = []

    # Iterating over the data
    iter_count = 1
    for inputs, label, attention_mask in tqdm(data_loader):
        inputs = inputs.long().to(args.device)
        attention_mask = attention_mask.int().to(args.device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            result = model(inputs, attention_mask)
            logits = result["logits"]
            targets = result["targets"]
            _, preds = torch.max(logits, 1)

            loss = criterion(logits, targets)
            # print('NCE: {}'.format(loss))

            # If there other losses available, i.e., kmeans loss
            if "kmeans_loss" in result:
                p = result["kmeans_loss"].float()
                loss += p
                print("kmeans: {}, both: {}".format(p, loss))

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            iter_count += 1

        # Appending predictions and loss
        running_loss += loss.item() * inputs.size(0)
        iter_acc.append(
            accuracy_score(targets.cpu().data.numpy(), preds.cpu().data.numpy())
        )

        # Deeper tracking: code perplexity and losses per iteration
        phase = "train"
        if iter_count % 50 == 0:
            wandb.log(
                {
                    "Code perplexity/"
                    + phase: result["code_perplexity"].cpu().data.numpy(),
                    "Iter loss/" + phase: loss.item(),
                    "Iter kmeans loss/"
                    + phase: result["kmeans_loss"].float().cpu().data.numpy()
                    if args.vq_type == "kmeans"
                    else 0.0,
                    "Iter acc/" + phase: iter_acc[-1],
                    "iter_count": iter_count,
                }
            )

    # Statistics
    loss = running_loss / dataset_size
    print("Train Iter acc: {}, loss: {}".format(iter_acc, loss))

    update_loss(
        phase="train",
        running_meter=running_meter,
        loss=loss,
        accuracy=np.mean(iter_acc),
        epoch=epoch,
        wandb=wandb,
    )

    return model, optimizer, scheduler


def evaluate(
    model, data_loader, criterion, args, epoch, phase, dataset_size, running_meter
):
    model.eval()

    # To track the loss and other metrics
    running_loss = 0.0
    iter_acc = []

    # Iterating over the data
    iter_count = 1

  


    for  inputs, label, attention_mask  in tqdm(data_loader):
        
        inputs = inputs.long().to(args.device)
        attention_mask = attention_mask.int().to(args.device)

        with torch.set_grad_enabled(False):
            result =  model(inputs, attention_mask)
            logits = result["logits"]
            targets = result["targets"]
            _, preds = torch.max(logits, 1)

            loss = criterion(logits, targets)
            print("NCE: {}".format(loss))

            # If there other losses available, i.e., kmeans loss
            if "kmeans_loss" in result:
                p = result["kmeans_loss"].float()
                loss += p
                print("kmeans: {}, both: {}".format(p, loss))

            iter_count += 1

        # Appending predictions and loss
        running_loss += loss.item() * inputs.size(0)
        iter_acc.append(
            accuracy_score(targets.cpu().data.numpy(), preds.cpu().data.numpy())
        )

        if iter_count % 50 == 0:
            wandb.log(
                {
                    "Code perplexity/"
                    + phase: result["code_perplexity"].cpu().data.numpy(),
                    "Iter loss/" + phase: loss.item(),
                    "Iter kmeans loss/"
                    + phase: result["kmeans_loss"].float().cpu().data.numpy()
                    if args.vq_type == "kmeans"
                    else 0.0,
                    "Iter acc/" + phase: iter_acc[-1],
                }
            )

    # Statistics
    loss = running_loss / dataset_size
    print("Val Iter acc: {}, loss: {}".format(iter_acc, loss))

    update_loss(
        phase=phase,
        running_meter=running_meter,
        loss=loss,
        accuracy=np.mean(iter_acc),
        epoch=epoch,
        wandb=wandb,
    )

    return
