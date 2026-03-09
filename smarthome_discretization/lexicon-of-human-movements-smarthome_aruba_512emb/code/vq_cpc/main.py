import socket
import os
import numpy as np

from functools import partial

from ray import tune, air
from ray.tune import CLIReporter

# from ray.tune.suggest.basic_variant import BasicVariantGenerator

from arguments import parse_args
from trainer import learn_model
from utils import set_all_seeds
from evaluate_with_classifier import evaluate_with_classifier


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    set_all_seeds(args.random_seed)
    print(args)

    # EXPERIMENT: Final final pretraining runs on RTX 6000.
    # Going back to the old environment itself, but this is with cosine
    # scheduling.
    # Now varying the input downsampling and seeing if the performance falls
    data_perc = [100]
    hostname = socket.gethostname()

    config = {
        "data_perc": tune.choice(data_perc),
        "random_seed": tune.choice([42]),
        "aggregator": tune.choice(['conv']),
        "vq_type":  tune.choice(["kmeans"]),
        "vq_gamma": tune.choice([0.25]),
        "num_conv_agg_layers": tune.choice([2,4,6]),
        "batch_size":tune.choice([64,128]),
        "num_steps_prediction": tune.choice([10,12]),
        "num_negatives": tune.choice([10,15]),
        "learning_rate": tune.choice([1e-3,1e-4,5e-4]),
        "weight_decay": tune.choice([0.0,1e-4,1e-5]),
        "window": tune.choice([20]),
        "caption_type":tune.choice(["numerical_values"]),
       
        # "input_downsampling": tune.grid_search([2]),
        "pre_exp_name": tune.sample_from(
            lambda spec: '{}'.format(hostname)),
        # "classifier_lr": tune.choice([1e-3, 1e-4, 5e-4]),
        # "classifier_wd": tune.choice([0.0, 1e-4, 1e-5]),
        # "classifier_batch_size": tune.choice([128, 256]),
    }

    # num_samples = 10

    num_samples =10
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(learn_model),
            resources={"cpu": 8, "gpu": 0.2},
        ),
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
        ),
        param_space=config,
        run_config=air.RunConfig(
            storage_path=os.path.join(dir_path, "ray_results")
        ),
    )
    results = tuner.fit()

    # config={}
    # learn_model(config)
    print("---------Training complete!---------")
