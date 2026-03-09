import matplotlib
import numpy as np
import os
from functools import partial
from trainer import learn_model
from ray import tune
from ray.tune import CLIReporter

from arguments import parse_args
from utils import set_all_seeds

matplotlib.use("Agg")

# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Pre-training with Ray")
    args = parse_args()
    set_all_seeds(args.random_seed)
    print(args)

    # Experiment: first run of the quantization experiment
    overall_dir = (
        "/srv/share/hharesamudram3/capture-24/code/data_preparation"
        "/capture_24/all_data/Dec-31-2021"
    )
    # overall_dir = "/coc/pcba1/hharesamudram3/capture_24/code/data_preparation" \
    #               "/capture_24/all_data/Dec-31-2021"
    learning_rate = [0.0001, 0.0003, 0.0005]
    weight_decay = [0.0, 0.0001, 0.0005]
    quantization_method = ["kmeans", "gumbel"]
    data_perc = [1, 10.0]
    # data_perc = [0.01]

    config = {
        "learning_rate": tune.grid_search(learning_rate),
        "weight_decay": tune.grid_search(weight_decay),
        "data_perc": tune.grid_search(data_perc),
        "quantization_method": tune.grid_search(quantization_method),
        "root_dir": os.path.join(overall_dir, "capture_24_sr_50_users_100"),
        "pre_exp_name": tune.sample_from(
            lambda spec: "ultron-data-frac-{}".format(spec.config.data_perc)
        ),
    }

    num_samples = 1
    reporter = CLIReporter(
        metric_columns=["train_loss", "val_loss", "training_iteration"]
    )

    result = tune.run(
        partial(learn_model, args=args),
        resources_per_trial={"cpu": 2, "gpu": 0.5},
        config=config,
        num_samples=num_samples,
        progress_reporter=reporter,
        local_dir="./ray_results",
    )

    print("------ Pre-training complete! ------")
