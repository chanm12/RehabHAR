import os
import socket


def get_dataset_locs(args):
    hostname = socket.gethostname()
    if hostname == "zatopek":
        folder_name = "/coc/pcba1/hharesamudram3/capture_24"
    else:
        folder_name = "/srv/share/hharesamudram3/capture-24"

    # Whether to load unnormalized data
    unnormalized = ""
    if args.unnormalized == "True":
        unnormalized = "unnormalized"

    # Training fold
    fold = args.fold

    # Capture 24
    if args.dataset == "capture_24":
        args.root_dir = os.path.join(
            folder_name, "code/data_preparation/capture_24/all_data/Dec-20-2021"
        )
        args.data_file = "capture_24_sr_50.joblib"

    # If we want only the scaler
    scaler = os.path.join(
        folder_name,
        "code/data_preparation/capture_24/all_data/Dec-20" "-2021",
        "scaler.joblib",
    )

    # Waist
    if args.dataset == "mobiact":
        args.root_dir = os.path.join(
            folder_name,
            "code/data_preparation/mobiactv2/all_data/Oct-10-2021",
            unnormalized,
        )
        args.data_file = "mobiactv2_6_sr_50_fold_{}.joblib".format(fold)
        args.num_classes = 11
    if args.dataset == "motionsense":
        args.root_dir = os.path.join(
            folder_name,
            "code/data_preparation/motionsense/all_data/Oct-10-2021",
            unnormalized,
        )
        args.data_file = "motionsense_3_sr_50_fold_{}.joblib".format(fold)
        args.num_classes = 6
    if args.dataset == "usc_had":
        args.root_dir = os.path.join(
            folder_name,
            "code/data_preparation/usc_had/all_data/Nov-27-2021",
            unnormalized,
        )
        args.data_file = "usc_had_3_sr_50_fold_{}.joblib".format(fold)
        args.num_classes = 12

    # Arms
    if args.dataset == "hhar":
        args.root_dir = os.path.join(
            folder_name, "code/data_preparation/hhar/all_data/Nov-24-2021", unnormalized
        )
        args.data_file = "hhar_watch_sr_50_fold_{}.joblib".format(fold)
        args.num_classes = 6
    if args.dataset == "myogym":
        args.root_dir = os.path.join(
            folder_name,
            "code/data_preparation/myogym/all_data/Nov-24-2021",
            unnormalized,
        )
        args.data_file = "myogym_3_sr_50_fold_{}.joblib".format(fold)
        args.num_classes = 31
    if args.dataset == "wetlab":
        args.root_dir = os.path.join(
            folder_name,
            "code/data_preparation/wetlab/all_data/Nov-26-2021",
            unnormalized,
        )
        args.data_file = "wetlab_3_sr_50_fold_{}.joblib".format(fold)
        args.num_classes = 9

    # Legs/ankle
    if args.dataset == "pamap":
        args.root_dir = os.path.join(
            folder_name,
            "code/data_preparation/pamap2/all_data/Nov-27-2021",
            unnormalized,
        )
        args.data_file = "pamap2_3_sr_50_fold_{}.joblib".format(fold)
        args.num_classes = 12
    if args.dataset == "daphnet":
        args.root_dir = os.path.join(
            folder_name,
            "code/data_preparation/daphnet/all_data/Nov-25-2021",
            unnormalized,
        )
        args.data_file = "daphnet_sr_64_fold_{}.joblib".format(fold)
        args.num_classes = 3
    if args.dataset == "mhealth":
        args.root_dir = os.path.join(
            folder_name,
            "code/data_preparation/mhealth/all_data/Nov-25-2021",
            unnormalized,
        )
        args.data_file = "mhealth_3_sr_50_fold_{}.joblib".format(fold)
        args.num_classes = 13

    print(
        "The dataset name is: {}, loc is: {}".format(
            args.dataset, os.path.join(args.root_dir, args.data_file)
        )
    )

    return args, scaler
