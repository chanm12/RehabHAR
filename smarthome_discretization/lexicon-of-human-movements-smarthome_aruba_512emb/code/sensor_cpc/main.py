import socket

from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from arguments import parse_args
from trainer import learn_model
from utils import set_all_seeds

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    set_all_seeds(args.random_seed)
    print(args)

    # EXPERIMENT: Doing the pre-training on sensor data (without any
    # quantization) to see how the training looks like.
    # # Increasing the data size but not aggregator because somehow many runs
    # fail.
    # data_perc = [100] # [100]
    # hostname = socket.gethostname()

    # num_steps_prediction = [10, 10, 10, 10, 10, 10, 12, 12, 10, 10, 10, 10]
    # learning_rate = [0.0005, 0.0005, 0.0005, 0.0001, 0.0005, 0.0005, 5e-05, 5e-05, 0.0001, 0.0005, 0.0005, 0.0005]
    # weight_decay = [0.0001, 0.0, 0.0001, 0.0, 0.0001, 1e-05, 0.0, 0.0, 1e-05, 0.0001, 0.0001, 0.0]
    # num_negatives = [10, 10, 10, 10, 10, 15, 10, 10, 15, 10, 10, 10]

    # # increasing the number of layers here
    # num_conv_agg_layers = [2, 6, 2, 6, 2, 4, 6, 6, 4, 2, 2, 6]

    # comb = [1, 3, 5, 7, 9, 11]

    # config = {
    #     "comb": tune.grid_search(
    #         comb),
    #     "data_perc": tune.grid_search(
    #         data_perc),

    #     "random_seed": tune.grid_search([42]),
    #     "input_downsampling": tune.grid_search([2]),  # downsampling input

    #     "aggregator": 'conv',
    #     "num_conv_agg_layers": tune.sample_from(
    #         lambda spec: num_conv_agg_layers[spec.config.comb]),

    #     "num_steps_prediction": tune.sample_from(
    #         lambda spec: num_steps_prediction[spec.config.comb]),
    #     "num_negatives": tune.sample_from(
    #         lambda spec: num_negatives[spec.config.comb]),
    #     "learning_rate": tune.sample_from(
    #         lambda spec: learning_rate[spec.config.comb]),
    #     "weight_decay": tune.sample_from(
    #         lambda spec: weight_decay[spec.config.comb]),

    #     "pre_exp_name": tune.sample_from(
    #         lambda spec: '{}-data-frac-big-{}'.format(hostname,
    #                                               spec.config.data_perc)),
    # }

    # num_samples = 1
    # reporter = CLIReporter(
    #     metric_columns=["train_loss", "val_loss", "train_acc", "val_acc",
    #                     "training_iteration"])

    # result = tune.run(
    #     partial(learn_model, args=args),
    #     resources_per_trial={"cpu": 3, "gpu": 0.5},
    #     config=config,
    #     search_alg=BasicVariantGenerator(constant_grid_search=True),
    #     num_samples=num_samples,
    #     progress_reporter=reporter,
    #     local_dir="./ray_results")

    learn_model(args)

    print("---------Training complete!---------")
