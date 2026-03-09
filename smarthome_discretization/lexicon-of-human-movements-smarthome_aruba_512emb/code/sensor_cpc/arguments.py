import argparse

import os
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Parameters for the Quantization")

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
    parser.add_argument(
        "--data_file", type=str, default="capture_24_debug_sr_50.joblib"
    )
    parser.add_argument("--num_classes", type=int, default=14)

    # -----------------------------------------------------------

    # Training settings
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
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
    parser.add_argument(
        "--data_perc",
        type=int,
        default=100,
        help="Percentage of the training windows to utilize",
    )

    # ------------------------------------------------------------

    # CPC parameters
    parser.add_argument(
        "--aggregator_type",
        type=str,
        default="conv",
        choices=["conv", "gru"],
        help="Choosing between the multi-layer conv "
        "aggregator or a GRU (like in vanilla CPC)",
    )
    parser.add_argument(
        "--num_steps_prediction",
        type=int,
        default=12,
        help="Number of steps in the future to predict",
    )
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=10,
        help="Number of negatives for the contrastive loss",
    )
    parser.add_argument(
        "--conv_feature_layers",
        type=str,
        default="[(32, 4, 2), (64, 4, 2), (128, 1, 1), " "(256, 1, 1)]",
    )
    parser.add_argument(
        "--conv_aggregator_layers",
        type=str,
        default="[(256, 2, 1), (256, 3, 1), (256, 4, 1), "
        "(256, 5, 1), (256, 6, 1), (256, 7, 1), "
        "(256, 8, 1), (256, 9, 1), (256, 10, 1), "
        "(256, 11, 1), (256, 12, 1)]",
    )
    parser.add_argument(
        "--num_conv_agg_layers",
        type=int,
        default=5,
        help="The number of conv aggregator layers to use",
    )
    parser.add_argument(
        "--input_downsampling",
        type=int,
        default=4,
        choices=[2, 4],
        help="The factor by which the input will be "
        "downsampled by the encoder architecture using "
        "stride. When using 4, it has 2 layers with "
        "filter = 4, stride =2. When it is 2, one layer "
        "is used with stride while the rest are conv 1.",
    )

    # ------------------------------------------------------------

    # Classification parameters
    parser.add_argument(
        "--classifier_lr",
        type=float,
        default=1e-4,
        help="Learning rate for the classifier",
    )
    parser.add_argument(
        "--classifier_wd", type=float, default=0.0, help="L2 norm for the classifier"
    )
    parser.add_argument(
        "--classifier_batch_size",
        type=int,
        default=256,
        help="Batch size for the classifier",
    )
    parser.add_argument(
        "--saved_model_folder",
        type=str,
        default=None,
        help="Folder path of the learned CPC model",
    )
    parser.add_argument(
        "--learning_schedule",
        type=str,
        default="last_layer",
        choices=["last_layer", "all_layers"],
        help="whether to train all layers or the last layer",
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
        choices=["True", "False"],
        help="For UCI-HAR, certain batch sizes dont work "
        "since they result in 1 sample. So can drop that",
    )
    parser.add_argument(
        "--unnormalized",
        type=str,
        default="True",
        help="if we need to load unnormalized target datasets",
    )
    parser.add_argument(
        "--encoder_only",
        type=str,
        default="False",
        help="Whether to only transfer the encoder weights "
        "from pre-training to classification or not",
    )
    parser.add_argument(
        "--classification_model",
        type=str,
        default="linear",
        choices=["linear", "mlp"],
        help="Choosing the classifier: linear is a single FC "
        "layer whereas MLP is the 3-layer network with "
        "BN and ReLU and dropout.",
    )

    # ------------------------------------------------------------

    # Mutliple runs
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
        help="Adding an experiment name for pre-training",
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
