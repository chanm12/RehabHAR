import matplotlib
import numpy as np
from functools import partial
from ray import tune
from ray.tune import CLIReporter

from arguments import parse_args
from quantize_data import generate_quantized_data
from utils import set_all_seeds

matplotlib.use("Agg")

# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Quantization with Ray")
    args = parse_args()
    set_all_seeds(args.random_seed)
    print(args)

    # EXPERIMENT: quantizing data from the kmeans pre-training where I did a
    # new param sweep. This is the first half of the sweep!
    data_perc = [10]
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

    # Params from the second half of the sweep
    num_steps_prediction = [10, 12]
    num_negatives = [10, 15]
    learning_rate = [1e-04, 1e-05]
    weight_decay = [1e-05]

    num_conv_agg_layers = [2, 4]
    vq_gamma = [0.1, 0.2]

    comb = list(np.arange(len(num_steps_prediction)))

    config = {
        "data_perc": tune.grid_search(data_perc),
        "dataset": tune.grid_search(dataset),
        "fold": tune.grid_search([0, 1, 2, 3, 4]),
        "random_seed": tune.grid_search([42]),
        "input_downsampling": tune.grid_search([2]),  # downsampling input
        "aggregator": "conv",
        "vq_type": "kmeans",
        "vq_gamma": tune.grid_search(vq_gamma),
        "num_conv_agg_layers": tune.grid_search(num_conv_agg_layers),
        "num_steps_prediction": tune.grid_search(num_steps_prediction),
        "num_negatives": tune.grid_search(num_negatives),
        "learning_rate": tune.grid_search(learning_rate),
        "weight_decay": tune.grid_search(weight_decay),
        "pre_exp_name": tune.sample_from(
            lambda spec: "{}-data-frac-{}".format("sophon", spec.config.data_perc)
        ),
        "saved_model_folder": "collected_sophon_Mar_26_2022",
    }

    num_samples = 1
    reporter = CLIReporter(metric_columns=["done", "training_iteration"])

    result = tune.run(
        partial(generate_quantized_data, args=args),
        resources_per_trial={"cpu": 2, "gpu": 0.5},
        config=config,
        num_samples=num_samples,
        progress_reporter=reporter,
        local_dir="./ray_results",
    )

    print("------ Pre-training complete! ------")
