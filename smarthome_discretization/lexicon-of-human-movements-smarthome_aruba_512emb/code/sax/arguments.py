import argparse

import os
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parameters for the creating SAX features and "
        "classifying on them"
    )

    # SAX computation parameters
    parser.add_argument(
        "--dataset", type=str, default="mobiact", help="Dataset to compute features for"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Window size on which the SAX compuatation is " "performed",
    )
    parser.add_argument(
        "--overlap", type=int, default=50, help="Overlap between consecutive windows"
    )
    parser.add_argument(
        "--span",
        type=int,
        default=2,
        help="The number of time steps each symbol will " "correspond to",
    )
    parser.add_argument(
        "--num_sax_symbols",
        type=int,
        default=512,
        help="The total number of symbols SAX can compute. "
        "Typically it is lower than this number",
    )
    parser.add_argument(
        "--unnormalized",
        type=str,
        default="False",
        help="if we need to load unnormalized target datasets",
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
        "--embedding_size",
        type=int,
        default=128,
        help="Dimension size of the NLP-like embeddings",
    )
    parser.add_argument("--num_epochs", type=int, default=50)

    # ------------------------------------------------------------
    # Other params
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="Selecting the GPU to execute it with"
    )
    parser.add_argument(
        "--random_seed", type=int, choices=[5, 10, 20, 30, 40, 42], default=42
    )
    parser.add_argument(
        "--exp_name",
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
