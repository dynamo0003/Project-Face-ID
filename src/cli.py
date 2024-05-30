import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A CLI interface for our face recognition model",
    )

    parser.add_argument("model", help="path to the model data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-t",
        "--train",
        metavar="DATASET",
        help="train the model on a dataset",
    )
    group.add_argument(
        "-i",
        "--image",
        metavar="IMAGE",
        help="evaluate an image",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="COUNT",
        type=int,
        default=10,
        help="specify the number of epochs",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="SIZE",
        type=int,
        default=100,
        help="specify the batch size",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        metavar="RATE",
        type=float,
        default=0.001,
        help="specify the learning rate",
    )
    parser.add_argument(
        "-c",
        "--cpu",
        action="store_true",
        help="use the CPU",
    )
    parser.add_argument(
        "-w",
        "--warnings",
        action="store_true",
        help="show warnings",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    import warnings

    from model import Model

    if not args.warnings:
        warnings.filterwarnings("ignore")

    model = Model(4, use_cpu=args.cpu)
    if args.train:
        model.train(
            args.train,
            args.epochs,
            args.batch_size,
            args.learning_rate,
        )
        model.save(args.model)
    else:
        model.load(args.model)
        model.eval(args.image)
