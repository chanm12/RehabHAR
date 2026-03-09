import matplotlib
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from arguments import parse_args
from evaluate_with_classifier import evaluate_with_classifier
from utils import set_all_seeds

matplotlib.use("Agg")

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Training the classifiers with ray")
    args = parse_args()
    set_all_seeds(args.random_seed)

    # EXPERIMENT: training a simple RNN classifier on the SAX quantized dataset
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

    classifier_lr = [0.0001, 0.0003, 0.0005]
    classifier_wd = [0.0, 0.0001, 0.0005]

    config = {
        "dataset": tune.grid_search(dataset),
        "fold": tune.grid_search([0]),
        "classifier_lr": tune.grid_search(classifier_lr),
        "classifier_wd": tune.grid_search(classifier_wd),
        "date": "Mar-05-2022",
        "exp_name": "ultron",
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
