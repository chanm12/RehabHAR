import matplotlib
from functools import partial
from ray import tune
from ray.tune import CLIReporter

from arguments import parse_args
from compute_features import compute_sax_features
from utils import set_all_seeds

matplotlib.use("Agg")

# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Computing SAX features with Ray!")
    args = parse_args()
    set_all_seeds(args.random_seed)
    print(args)

    # EXPERIMENT: computing the SAX features for all target datasets for easy
    # and direct later access
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

    config = {
        "dataset": tune.grid_search(dataset),
        "fold": tune.grid_search([0, 1, 2, 3, 4]),
    }

    num_samples = 1
    reporter = CLIReporter(metric_columns=["status", "training_iteration"])

    result = tune.run(
        partial(compute_sax_features, args=args),
        resources_per_trial={"cpu": 2},
        config=config,
        num_samples=num_samples,
        progress_reporter=reporter,
        local_dir="./ray_results",
    )

    print("------ Pre-training complete! ------")
