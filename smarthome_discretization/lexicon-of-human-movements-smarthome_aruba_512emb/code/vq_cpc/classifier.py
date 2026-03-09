import socket

import matplotlib
import numpy as np
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from arguments import parse_args
from evaluate_with_classifier import evaluate_with_classifier
from utils import set_all_seeds, get_git_revision_hash

matplotlib.use("Agg")

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Training the classifiers with ray")
    args = parse_args()
    set_all_seeds(args.random_seed)
    hostname = socket.gethostname()

    # Adding the git commit hash into the args
    args.commit_hash = get_git_revision_hash()
    assert args.commit_hash is not None

    # EXPERIMENT: VARYING THE INPUT DOWN-SAMPLING TO SEE HOW WIDE A SPAN IS
    # GOOD FOR DISCRETE REPRESENTATION LEARNING
    # Now that we can get the results as expected from claptrap
    # with the cuBLAS fix

    data_perc = [10]
    dataset = ["hhar", "mhealth", "mobiact", "motionsense", "myogym", "pamap"]
    fold = [0, 1, 2, 3, 4]

    input_type = ["discrete"]
    quant_date = ["Nov-12-2022"]  # need to verify this based on the node itself
    learning_schedule = ["all_layers"]

    num_steps_prediction = [10]
    num_negatives = [10]
    learning_rate = [0.0001]
    weight_decay = [0.0001]
    num_conv_agg_layers = [2]

    vq_gamma = [0.25]

    # Varying the input downsampling over here
    input_downsampling = [1, 4]

    # best performing classification params
    classifier_lr = [0.0005, 0.0005, 0.001, 0.0005, 0.0005, 0.001]
    classifier_wd = [0.0001, 1e-05, 0.0001, 0.0, 1e-05, 0.0001]
    comb = list(np.arange(len(classifier_lr)))
    random_seed = [10, 20, 30, 40, 42]

    node = "claptrap"

    config = {
        "comb": tune.grid_search(comb),
        "data_perc": tune.grid_search(data_perc),
        "dataset": tune.sample_from(lambda spec: dataset[spec.config.comb]),
        "fold": tune.grid_search(fold),
        "random_seed": tune.grid_search(random_seed),
        # downsampling input
        "input_downsampling": tune.grid_search(input_downsampling),
        "aggregator": "conv",
        "learning_schedule": tune.grid_search(learning_schedule),
        "input_type": tune.grid_search(input_type),
        "quant_date": tune.grid_search(quant_date),
        # pre-training params
        "vq_type": "kmeans",
        "vq_gamma": tune.choice(vq_gamma),
        "num_conv_agg_layers": tune.grid_search(num_conv_agg_layers),
        "num_steps_prediction": tune.grid_search(num_steps_prediction),
        "num_negatives": tune.grid_search(num_negatives),
        "learning_rate": tune.grid_search(learning_rate),
        "weight_decay": tune.grid_search(weight_decay),
        "classifier_lr": tune.sample_from(lambda spec: classifier_lr[spec.config.comb]),
        "classifier_wd": tune.sample_from(lambda spec: classifier_wd[spec.config.comb]),
        "classifier_type": "gru",
        "pre_exp_name": tune.sample_from(lambda spec: "{}".format(hostname)),
        "exp_name": tune.sample_from(
            lambda spec: "{}-input-downsampling-five-rs".format(hostname)
        ),
        # "saved_model_folder": "collected_pops_tars_Mar_15_2022",
    }
    num_samples = 1
    reporter = CLIReporter(
        metric_columns=["loss", "val_f1_score", "test_f1_scorhae", "training_iteration"]
    )

    result = tune.run(
        partial(evaluate_with_classifier, args=args),
        resources_per_trial={"cpu": 2, "gpu": 0.25},
        config=config,
        search_alg=BasicVariantGenerator(constant_grid_search=True),
        num_samples=num_samples,
        progress_reporter=reporter,
        local_dir="./ray_results",
    )

    print("------ Evaluation complete! ------")
