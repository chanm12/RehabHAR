import socket

import matplotlib
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

    # EXPERIMENT: Doing the classification after pre-training on sensor data (
    # without any quantization) to see how the training looks like.
    # Doing the five randomized runs for the models which have used 50 and
    # 100% of the pre-training data instead. The pre-train params are the
    # same as 10% (to save time)
    data_perc = [50]
    dataset = [
        "hhar",
        "hhar",
        "mhealth",
        "mhealth",
        "mobiact",
        "mobiact",
        "motionsense",
        "motionsense",
        "myogym",
        "myogym",
        "pamap",
        "pamap",
    ]
    fold = [0, 1, 2, 3, 4]
    learning_schedule = ["last_layer"]

    # pre-training params
    num_steps_prediction = [10, 10, 10, 10, 10, 10, 12, 12, 10, 10, 10, 10]
    learning_rate = [
        0.0005,
        0.0005,
        0.0005,
        0.0001,
        0.0005,
        0.0005,
        5e-05,
        5e-05,
        0.0001,
        0.0005,
        0.0005,
        0.0005,
    ]
    weight_decay = [
        0.0001,
        0.0,
        0.0001,
        0.0,
        0.0001,
        1e-05,
        0.0,
        0.0,
        1e-05,
        0.0001,
        0.0001,
        0.0,
    ]
    num_negatives = [10, 10, 10, 10, 10, 15, 10, 10, 15, 10, 10, 10]
    num_conv_agg_layers = [2, 6, 2, 6, 2, 4, 6, 6, 4, 2, 2, 6]
    comb = [1, 3, 5, 7, 9, 11]

    classifier_lr = [0.001, 0.0005, 0.0001]
    classifier_wd = [1e-05, 0.0, 1e-04]
    random_seed = [10, 20, 30, 40, 42]

    node = ["clank"] * len(comb)

    config = {
        "comb": tune.grid_search(comb),
        "data_perc": tune.grid_search(data_perc),
        "fold": tune.grid_search(fold),
        "random_seed": tune.grid_search(random_seed),
        "dataset": tune.sample_from(lambda spec: dataset[spec.config.comb]),
        "input_downsampling": tune.grid_search([2]),  # downsampling input
        "aggregator": "conv",
        "learning_schedule": tune.grid_search(learning_schedule),
        # pre-training params
        "num_conv_agg_layers": tune.sample_from(
            lambda spec: num_conv_agg_layers[spec.config.comb]
        ),
        "num_steps_prediction": tune.sample_from(
            lambda spec: num_steps_prediction[spec.config.comb]
        ),
        "num_negatives": tune.sample_from(lambda spec: num_negatives[spec.config.comb]),
        "learning_rate": tune.sample_from(lambda spec: learning_rate[spec.config.comb]),
        "weight_decay": tune.sample_from(lambda spec: weight_decay[spec.config.comb]),
        "classifier_lr": tune.grid_search(classifier_lr),
        "classifier_wd": tune.grid_search(classifier_wd),
        "classification_model": tune.grid_search(["mlp"]),
        "pre_exp_name": tune.sample_from(
            lambda spec: "{}-data-frac-big-{}".format(hostname, spec.config.data_perc)
        ),
        # 2-5 siri, 3-4 rosie, 1-6 vector
        "exp_name": tune.sample_from(
            lambda spec: "{}-data-frac-big-sweep-{}".format(
                hostname, spec.config.data_perc
            )
        ),
        "saved_model_folder": "collected_clank_sensor_cpc_big_Oct_09_2022",
    }

    num_samples = 1
    reporter = CLIReporter(
        metric_columns=["loss", "val_f1_score", "test_f1_score", "training_iteration"]
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
