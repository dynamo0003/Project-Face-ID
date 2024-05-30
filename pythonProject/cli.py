import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A CLI interface for our face recognition model",
    )

    parser.add_argument("model", help="path to the trained model data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-e", "--eval", metavar="IMAGE", help="evaluate an image")
    group.add_argument("-t", "--train", action="store_true", help="train the model on a dataset")
    parser.add_argument("-d", "--dataset", metavar="DATASET", help="specify the dataset folder")

    args = parser.parse_args()
    if args.train and not args.dataset:
        parser.error("the --dataset argument is required when training")
    return args


if __name__ == "__main__":
    args = parse_args()

    from model import Model

    model = Model(4)
    if args.train:
        model.train(args.dataset, 10, 32, 0.001)
        model.save(args.model)
    else:
        model.load(args.model)
        model.eval(args.eval)
