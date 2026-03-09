import socket

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

    # EXPERIMENT: VARYING THE INPUT DOWN-SAMPLING TO SEE HOW WIDE A SPAN IS
    # GOOD FOR DISCRETE REPRESENTATION LEARNING
    data_perc = [10]
    hostname = socket.gethostname()

    dataset = ["hhar", "mhealth", "mobiact", "motionsense", "myogym", "pamap"]
    fold = [0, 1, 2, 3, 4]

    # Params for the sweeping
    num_steps_prediction = [10]
    num_negatives = [10]
    learning_rate = [1e-4]
    weight_decay = [1e-4]
    num_conv_agg_layers = [2]

    vq_gamma = [0.25]
    node = "glados"

    input_downsampling = [1, 4]
    # comb = list(np.arange(len(num_steps_prediction)))

    config = {
        # "comb": tune.grid_search(comb),
        "data_perc": tune.grid_search(data_perc),
        "dataset": tune.grid_search(dataset),
        "fold": tune.grid_search(fold),
        "random_seed": tune.grid_search([42]),
        "input_downsampling": tune.grid_search(input_downsampling),
        # downsampling input
        "aggregator": "conv",
        "vq_type": "kmeans",
        "vq_gamma": tune.choice(vq_gamma),
        "num_conv_agg_layers": tune.grid_search(num_conv_agg_layers),
        "num_steps_prediction": tune.grid_search(num_steps_prediction),
        "num_negatives": tune.grid_search(num_negatives),
        "learning_rate": tune.grid_search(learning_rate),
        "weight_decay": tune.grid_search(weight_decay),
        "pre_exp_name": tune.sample_from(lambda spec: "{}".format(hostname)),
        "saved_model_folder": "collected_claptrap_vq_cpc_changed_input_downsample_Nov_11_2022",
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
