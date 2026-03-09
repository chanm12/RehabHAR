import matplotlib
import numpy as np
from finetuner import finetune
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from arguments import parse_args
from utils import set_all_seeds

matplotlib.use("Agg")

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Training the classifiers with ray")
    args = parse_args()
    set_all_seeds(args.random_seed)

    # Experiment: doing 5 random classifier runs with the best performing
    # pre-trained mnodel. The pre-training also utilizes 5 random states (the
    # same seeds are used here as well)
    comb = list(np.arange(9))
    dataset = [
        "daphnet",
        "hhar",
        "mhealth",
        "mobiact",
        "motionsense",
        "myogym",
        "pamap",
        "usc_had",
        "wetlab",
    ]
    learning_rate = [
        0.0003,
        0.0001,
        0.0003,
        0.0001,
        0.0003,
        0.0001,
        0.0001,
        0.0003,
        0.0003,
    ]
    weight_decay = [
        0.0005,
        0.0005,
        0.0005,
        0.0005,
        0.0005,
        0.0005,
        0.0005,
        0.0005,
        0.0001,
    ]
    classifier_lr = [
        0.0001,
        0.0003,
        0.0005,
        0.0003,
        0.0001,
        0.0003,
        0.0003,
        0.0005,
        0.0005,
    ]
    classifier_wd = [0.0001, 0.0, 0.0, 0.0005, 0.0001, 0.0005, 0.0, 0.0001, 0.0001]

    # Experiment: re-doing the parameter tuning because the runs are not
    # matching between experiments. Same experiment done two different ways
    # does not match. Therefore, doing once again.

    config = {
        "comb": tune.grid_search(comb),
        "fold": tune.grid_search([0, 1, 2, 3, 4]),
        "random_seed": tune.grid_search([10, 20, 30, 40, 42]),
        "dataset": tune.sample_from(lambda spec: dataset[spec.config.comb]),
        "learning_rate": tune.sample_from(lambda spec: learning_rate[spec.config.comb]),
        "weight_decay": tune.sample_from(lambda spec: weight_decay[spec.config.comb]),
        "classifier_lr": tune.sample_from(lambda spec: classifier_lr[spec.config.comb]),
        "classifier_wd": tune.sample_from(lambda spec: classifier_wd[spec.config.comb]),
        # "pre_exp_name": "vector-five-random",  # training with 5 random seeds
        "exp_name": "vector-five-random-no-pretrain",
        # "saved_model_folder": "collected_vector_Jan_20_2022",
    }

    num_samples = 1
    reporter = CLIReporter(
        metric_columns=["loss", "val_f1_score", "test_f1_score", "training_iteration"]
    )

    result = tune.run(
        partial(finetune, args=args),
        resources_per_trial={"cpu": 2, "gpu": 0.25},
        config=config,
        search_alg=BasicVariantGenerator(constant_grid_search=True),
        num_samples=num_samples,
        progress_reporter=reporter,
        local_dir="./ray_results",
    )

    print("------ Evaluation complete! ------")
