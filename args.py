# Copyright (c) EEEM071, University of Surrey
# Copyright (c) EEEM071, University of Surrey
import argparse


def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument(
        "--root", type=str, default="/content/drive/MyDrive/HMDB_simp", help="root path to data directory"
    )

    parser.add_argument(
        "--name", type=str, default="runs", help="Name for the summary on tesorboard"
    )
    # parser.add_argument(
    #     "-s",
    #     "--source-names",
    #     type=str,
    #     required=True,
    #     nargs="+",
    #     help="source dataset for training(delimited by space)",
    # )
    # parser.add_argument(
    #     "-t",
    #     "--target-names",
    #     type=str,
    #     required=True,
    #     nargs="+",
    #     help="target dataset for testing(delimited by space)",
    # )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        help="number of data loading workers (tips: 4 or 8 times number of gpus)",
    )
    parser.add_argument(
        "-sp",
        "--split",
        default=0.2,
        type=float,
        help="train/test split ratio",
    )
    parser.add_argument(
        "--mprint",
        action="store_true",
        default=False,
        help="Print Model Summary",
    )

    parser.add_argument(
        "-btch",
        "--batch",
        default=64,
        type=int,
        help="Batch size",
    )

    # ************************************************************
    # Optimization options
    # ************************************************************
    # parser.add_argument(
    #     "--optim",
    #     type=str,
    #     default="adam",
    #     help="optimization algorithm (see optimizers.py)",
    # )

    parser.add_argument(
        "--lr", default=0.0003, type=float, help="initial learning rate"
    )

    parser.add_argument(
        "--decay", default=1e-08, type=float, help="weight decay"
    )

    # # sgd
    # parser.add_argument(
    #     "--momentum",
    #     default=0.9,
    #     type=float,
    #     help="momentum factor for sgd and rmsprop",
    # )
    # parser.add_argument(
    #     "--sgd-dampening", default=0, type=float, help="sgd's dampening for momentum"
    # )
    # parser.add_argument(
    #     "--sgd-nesterov",
    #     action="store_true",
    #     help="whether to enable sgd's Nesterov momentum",
    # )
    # # rmsprop
    # parser.add_argument(
    #     "--rmsprop-alpha", default=0.99, type=float, help="rmsprop's smoothing constant"
    # )
    # # adam/amsgrad
    # parser.add_argument(
    #     "--adam-beta1",
    #     default=0.9,
    #     type=float,
    #     help="exponential decay rate for adam's first moment",
    # )
    # parser.add_argument(
    #     "--adam-beta2",
    #     default=0.999,
    #     type=float,
    #     help="exponential decay rate for adam's second moment",
    # )

    # ************************************************************
    # Training hyperparameters
    # ************************************************************
    parser.add_argument(
        "--epochs", default=20, type=int, help="number of epochs to run"
    )
    #
    # parser.add_argument(
    #     "--train-batch-size", default=32, type=int, help="training batch size"
    # )
    # parser.add_argument(
    #     "--test-batch-size", default=100, type=int, help="test batch size"
    # )

    # ************************************************************
    # Cross entropy loss-specific setting
    # # ************************************************************
    # parser.add_argument(
    #     "--label-smooth",
    #     action="store_true",
    #     help="use label smoothing regularizer in cross entropy loss",
    # )
    # parser.add_argument(
    #     "--lambda-xent",
    #     type=float,
    #     default=1,
    #     help="weight to balance cross entropy loss",
    # )

    # ************************************************************
    # Architecture
    # ************************************************************
    # parser.add_argument("-a", "--arch", type=str, default="resnet34")
    # parser.add_argument(
    #     "--no-pretrained", action="store_true", help="do not load pretrained weights"
    # )

    # ************************************************************
    # Test settings
    # ************************************************************

    # parser.add_argument("--evaluate", action="store_true", help="evaluate only")
    # parser.add_argument(
    #     "--eval-freq",
    #     type=int,
    #     default=-1,
    #     help="evaluation frequency (set to -1 to test only in the end)",
    # )
    # parser.add_argument(
    #     "--start-eval",
    #     type=int,
    #     default=0,
    #     help="start to evaluate after a specific epoch",
    # )
    # parser.add_argument(
    #     "--test_size",
    #     type=int,
    #     default=800,
    #     help="test-size for vehicleID dataset, choices=[800,1600,2400]",
    # )
    # ************************************************************
    # Miscs
    # ************************************************************
    # parser.add_argument("--print-freq", type=int, default=10, help="print frequency")
    parser.add_argument("--seed", type=int, default=42, help="manual seed")
    parser.add_argument(
        "--save-dir", type=str, default="/content/drive/MyDrive/TensorBoard_Logs/logstxt/", help="path to save log and model weights"
    )
    # parser.add_argument(
    #     "--gpu-devices",
    #     default="0",
    #     type=str,
    #     help="gpu device ids for CUDA_VISIBLE_DEVICES",
    # )
    return parser

#
# def dataset_kwargs(parsed_args):
#     """
#     Build kwargs for ImageDataManager in data_manager.py from
#     the parsed command-line arguments.
#     """
#     return {
#         "train_batch_size": parsed_args.train_batch_size,
#         "test_batch_size": parsed_args.test_batch_size,
#         "workers": parsed_args.workers,
#     }

#
# def optimizer_kwargs(parsed_args):
#     """
#     Build kwargs for optimizer in optimizers.py from
#     the parsed command-line arguments.
#     """
#     return {
#         "optim": parsed_args.optim,
#         "lr": parsed_args.lr,
#         "weight_decay": parsed_args.weight_decay,
#         "momentum": parsed_args.momentum,
#         "sgd_dampening": parsed_args.sgd_dampening,
#         "sgd_nesterov": parsed_args.sgd_nesterov,
#         "rmsprop_alpha": parsed_args.rmsprop_alpha,
#         "adam_beta1": parsed_args.adam_beta1,
#         "adam_beta2": parsed_args.adam_beta2,
#     }
