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
    model, dataset, classes, epochs, batch_size, learning_rate, loss_goal, cpu, warnings
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

    print("Evaluating image")
    evaluation = fr_model.eval(image)

    print(f"Probabilities:")
    for i in range(len(evaluation[1])):
        if platform.system() == "Linux":
            print("\033[92m" if i == evaluation[0] else "\033[91m", end="")
        print(f"{i}: {evaluation[1][i]}")


if __name__ == "__main__":
    cli()
