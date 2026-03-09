import argparse

import os
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parameters for the multi-task baseline"
    )

    # Data loading parameters
    parser.add_argument("-w", "--window", type=int, default=100, help="Window size")
    parser.add_argument("-op", "--overlap", type=int, default=0)
    parser.add_argument("--input_size", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="capture_24")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/coc/pcba1/hharesamudram3/capture_24/code/data_preparation/capture_24/all_data/Dec-20-2021",
    )
    parser.add_argument("--data_file", type=str, default="capture_24_sr_50.joblib")
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--debug", type=str, default=False)
    # -----------------------------------------------------------

    # Training settings
    parser.add_argument("-b", "--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="weight decay for the optimizer",
    )
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="Selecting the GPU to execute it with"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="number of epochs to wait before early stopping",
    )
    # -----------------------------------------------------------

    # Quantization parameters
    parser.add_argument(
        "--quantization_method",
        type=str,
        default="gumbel",
        choices=["gumbel", "kmeans"],
    )
    parser.add_argument(
        "--num_vars", type=int, default=100, help="Number of vectors for quantization"
    )
    parser.add_argument(
        "--groups", type=int, default=2, help="Number of groups for quantization"
    )

    # -----------------------------------------------------------

    # Classification parameters
    parser.add_argument(
        "--classifier_lr",
        type=float,
        default=3e-4,
        help="Learning rate for the classifier",
    )
    parser.add_argument(
        "--classifier_wd", type=float, default=1e-4, help="L2 norm for the classifier"
    )
    parser.add_argument(
        "--classifier_batch_size",
        type=int,
        default=256,
        help="Batch size for the classifier",
    )
    parser.add_argument(
        "--learning_schedule",
        type=str,
        default="last_layer",
        choices=["last_layer", "all_layers", "last_conv"],
        help="whether to train all layers or the last layer",
    )
    parser.add_argument(
        "--saved_model_folder",
        type=str,
        default=None,
        help="The pretrained model folder",
    )
    parser.add_argument(
        "--random_weights",
        type=str,
        default="False",
        help="If we wanna do end to end training with random "
        "weights instead of finetuning with learned ones",
    )
    parser.add_argument(
        "--few_shot",
        type=int,
        default=0,
        help="Number of samples per class for few shot " "learning",
    )
    parser.add_argument(
        "--drop_last",
        type=str,
        default="False",
        help="To drop the last batch if necessary",
    )
    parser.add_argument(
        "--unnormalized",
        type=str,
        default="True",
        help="if we need to load unnormalized target datasets",
    )

    # Target dataset fold
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="tracking the fold of the target dataset, " "for 5-fold evaluation",
    )

    # Random seed for reproducibility
    parser.add_argument(
        "--random_seed", type=int, choices=[5, 10, 20, 30, 40, 42], default=42
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Adding an experiment name for easy searching",
    )
    parser.add_argument(
        "--pre_exp_name",
        type=str,
        default="",
        help="Adding an experiment name for easy searching",
    )

    # ------------------------------------------------------------
    # Ray tune params
    parser.add_argument(
        "--ray-address",
        help="Address of Ray cluster for seamless distributed execution.",
        required=False,
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using " "Ray Client.",
    )

    # ------------------------------------------------------------

    args, _ = parser.parse_known_args()

    args.device = torch.device(
        "cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu"
    )

    return args
