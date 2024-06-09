import os
import platform
import warnings as warns

import click


@click.group(
    help="A CLI interface for our face recognition model",
    context_settings={"help_option_names": ["-h", "--help"]},
)
def cli():
    pass


@cli.command(
    help="Train the model on a dataset",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "-m",
    "--model",
    help="Path to save trained model to",
    required=True,
)
@click.option(
    "-d",
    "--dataset",
    help="Path to dataset folder",
    required=True,
)
@click.option(
    "-c",
    "--classes",
    help="Number of classes (Default: 2)",
    default=2,
)
@click.option(
    "-e",
    "--epochs",
    type=int,
    help="Number of epochs (Default: 10)",
    default=10,
)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    help="Batch size (Default: 100)",
    default=100,
)
@click.option(
    "-l",
    "--learning-rate",
    type=float,
    help="Learning rate (Default: 0.001)",
    default=0.001,
)
@click.option(
    "-L",
    "--loss-goal",
    type=float,
    help="If this loss is reached when training, it stops",
    default=None,
)
@click.option(
    "-C",
    "--cpu",
    help="Use CPU",
    is_flag=True,
)
@click.option(
    "-w",
    "--warnings",
    help="Show warnings",
    is_flag=True,
)
def train(
    model,
    dataset,
    classes,
    epochs,
    batch_size,
    learning_rate,
    loss_goal,
    cpu,
    warnings,
):
    from model import Model

    if not warnings:
        warns.filterwarnings("ignore")
    fr_model = Model(classes, cpu)
    fr_model.train(dataset, epochs, batch_size, learning_rate, loss_goal)
    fr_model.save(model)


@cli.command(
    help="Evaluate an image using a trained model",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "-m",
    "--model",
    help="Path to trained model",
    required=True,
)
@click.option(
    "-i",
    "--image",
    help="Path to image",
    required=True,
)
@click.option(
    "-c",
    "--classes",
    help="Number of classes (Default: 2)",
    default=2,
)
@click.option(
    "-C",
    "--cpu",
    help="Use CPU",
    is_flag=True,
)
@click.option(
    "-w",
    "--warnings",
    help="Show warnings",
    is_flag=True,
)
def eval(model, image, classes, cpu, warnings):
    from model import Model

    if not warnings:
        warns.filterwarnings("ignore")
    fr_model = Model(classes, cpu)
    fr_model.load(model)
    choice, threshold, probs = fr_model.eval(image)

    print(f"Threshold: {threshold:.3f}\nProbabilities:")
    for i, p in enumerate(probs):
        if platform.system() == "Linux":
            if i == choice:
                if p >= threshold:
                    print("\033[32m", end="")  # Green
                else:
                    print("\033[33m", end="")  # Yellow
            else:
                print("\033[31m", end="")  # Red
        print(f"{i}: {p:.3f}")


@cli.command(
    help="Test the accuraccy of a trained model",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "-m",
    "--model",
    help="Path to trained model",
    required=True,
)
@click.option(
    "-d",
    "--dataset",
    help="Path to dataset folder",
    required=True,
)
@click.option(
    "-c",
    "--classes",
    help="Number of classes (Default: 2)",
    default=2,
)
@click.option(
    "-C",
    "--cpu",
    help="Use CPU",
    is_flag=True,
)
@click.option(
    "-w",
    "--warnings",
    help="Show warnings",
    is_flag=True,
)
def test(model, dataset, classes, cpu, warnings):
    from model import Model

    if not warnings:
        warns.filterwarnings("ignore")
    fr_model = Model(classes, cpu)
    fr_model.load(model)

    for i, dir in enumerate(os.listdir(dataset)):
        count = 0
        imgs = os.listdir(os.path.join(dataset, dir))
        for j, img in enumerate(imgs):
            if fr_model.eval(os.path.join(dataset, dir, img))[0] == i:
                count += 1
            print(f"\r{dir} ({i}): {j}/{len(imgs)}", end="")
        print(f"\r{dir} ({i}): {count}/{len(imgs)} ({count / len(imgs) * 100:.0f}%)")


if __name__ == "__main__":
    cli()
