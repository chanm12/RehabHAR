from datetime import date
from time import time

import numpy as np
import os
import torch
import torch.nn as nn
import wandb
from ray import tune
from torch import optim
from torch.optim.lr_scheduler import StepLR

from smarthome_dataset import load_classifier_dataset, load_indexed_dataset
from dataset_locs import get_dataset_locs
from meter import RunningMeter, BestMeter
from model import Classifier, QuantizedClassifier
from utils import (
    save_meter,
    compute_best_metrics,
    compute_classifier_metrics,
    model_save_name,
    set_all_seeds,
    update_args,
)
from arguments import parse_args

# ------------------------------------------------------------------------------


def evaluate_with_classifier(config, args=None):
    args = parse_args()
    # print('Inside classifier')
    # print(args)

    # Adding the config params back into the arg parser
    args = update_args(config, args)
    args, _ = get_dataset_locs(args)
    # print('Updated args', args)

    # Setting seed after updating args in case the seed is updated
    set_all_seeds(args.random_seed)

    # Getting the trained model name
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args.saved_model_folder is not None:
        args.saved_model = os.path.join(dir_path, 'models',
                                        args.saved_model_folder,
                                        model_save_name(args, capture=True,
                                                        classifier=False) +
                                        '.pkl')
    else:
        args.saved_model = None

    print("saved model", args.saved_model)

    # initialize Wandb
    name = (
        date.today().strftime("%b-%d-%Y") + "_" + model_save_name(args, classifier=True)
    )
    wandb.init(
        config=args,
        project="lexicon_smarthome_textdata_classifier",
        name=name,
        settings=wandb.Settings(start_method="thread"),
        mode="disabled",
    )

    # Load the data
    if args.input_type == "sensor":
        data_loaders, dataset_sizes = load_classifier_dataset(args)

    elif args.input_type == "discrete":
        data_loaders, dataset_sizes = load_indexed_dataset(args)
        args.vocab_size = data_loaders["train"].dataset.vocab_size

    # Tracking meters
    running_meter = RunningMeter(args=args)
    best_meter = BestMeter()

    # Creating the model
    if args.input_type == "sensor":
        model = Classifier(args=args).to(args.device)
    else:
        model = QuantizedClassifier(args=args).to(args.device)
    wandb.watch(model)  # Tracking the model with wandb

    model.load_pretrained_weights(args)

    # Optimizer settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.classifier_lr, weight_decay=args.classifier_wd
    )
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

    for epoch in range(0, args.num_epochs):
        since = time()
        print("Epoch {}/{}".format(epoch, args.num_epochs - 1))
        print("-" * 10)

        # Training
        model, optimizer, scheduler = train(
            model,
            data_loaders["train"],
            criterion,
            optimizer,
            scheduler,
            args,
            epoch,
            dataset_sizes["train"],
            running_meter,
        )

        # Validation
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

        # print("sizes", dataset_sizes)

        # Saving the logs
        save_meter(args, running_meter, finetune=True)

        # Printing the time taken
        time_elapsed = time() - since
        print(
            "Epoch {} completed in {:.0f}m {:.0f}s".format(
                epoch, time_elapsed // 60, time_elapsed % 60
            )
        )

        # tune.report(loss=running_meter.loss["val"][-1],
        #             val_f1_score=running_meter.f1_score["val"][-1],
        #             test_f1_score=running_meter.f1_score["test"][-1])

    # Computing the best metrics
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
    model,
    data_loader,
    criterion,
    optimizer,
    scheduler,
    args,
    epoch,
    dataset_size,
    running_meter,
):
    # Setting the model to training mode
    model.train()

    # Set only softmax layer to trainable
    if args.learning_schedule == "last_layer" and args.input_type=="sensor":
        model.freeze_encoder_layers()

    # To track the loss and other metrics
    running_loss = 0.0
    actual_labels = []
    pred_labels = []

    # Iterating over the data
    for data in data_loader:
        
        if args.input_type == "discrete":
            inputs, labels =data
            inputs = inputs.long().to(args.device)
        else:
            inputs, labels , attention_mask = data
            
            inputs = inputs.long().to(args.device)
            attention_mask = attention_mask.int().to(args.device)

        labels = labels.long().to(args.device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            if args.input_type == "sensor":
                outputs = model(inputs, attention_mask)
            else:
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        # Appending predictions and loss
        running_loss += loss.item() * inputs.size(0)
        actual_labels.extend(labels.cpu().data.numpy())
        pred_labels.extend(preds.cpu().data.numpy())

    scheduler.step()

    # Statistics
    loss = running_loss / dataset_size
    _ = compute_classifier_metrics(
        actual_labels, pred_labels, "train", running_meter, loss, epoch, wandb
    )

    return model, optimizer, scheduler


def evaluate(
    model, data_loader, args, criterion, epoch, phase, dataset_size, running_meter
):
    # Setting the model to eval mode
    model.eval()

    # To track the loss and other metrics
    running_loss = 0.0
    actual_labels = []
    pred_labels = []

    # Iterating over the data
    for data in data_loader:
        if args.input_type == "discrete":
            inputs, labels = data
            inputs = inputs.long().to(args.device)
        else:
            inputs, labels, attention_mask = data
            inputs = inputs.long().to(args.device)
            attention_mask = attention_mask.int().to(args.device)

        labels = labels.long().to(args.device)

        with torch.set_grad_enabled(False):
            if args.input_type == "sensor":
                outputs = model(inputs, attention_mask)
            else:
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # Appending predictions and loss
        running_loss += loss.item() * inputs.size(0)
        actual_labels.extend(labels.cpu().data.numpy())
        pred_labels.extend(preds.cpu().data.numpy())

    # Statistics
    loss = running_loss / dataset_size
    _ = compute_classifier_metrics(
        actual_labels, pred_labels, phase, running_meter, loss, epoch, wandb
    )

    return


if __name__ == "__main__":
    args = parse_args()
    print(args)
    set_all_seeds(args.random_seed)
    evaluate_with_classifier(config={}, args=args)
