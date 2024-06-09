import sys

from click.testing import CliRunner

sys.path.append("src")

import cli


def test_cli_help():
    out1 = CliRunner().invoke(cli.cli)
    out2 = CliRunner().invoke(cli.cli, ["-h"])
    assert out1.exit_code == out2.exit_code == 0
    print(out1.output)
    assert out1.output == out2.output
    assert "Usage: cli [OPTIONS] COMMAND [ARGS]..." in out1.output
    assert "A CLI interface for our face recognition model" in out1.output
    assert "Options:" in out1.output
    assert "-h, --help  Show this message and exit." in out1.output
    assert "Commands:" in out1.output
    assert "eval   Evaluate an image using a trained model" in out1.output
    assert "train  Train the model on a dataset" in out1.output


def test_cli_train():
    out = CliRunner().invoke(cli.train)
    assert out.exit_code != 0
    assert "Usage: train [OPTIONS]" in out.output
    assert "Try 'train -h' for help." in out.output
    assert "Error: Missing option '-m' / '--model'." in out.output


def test_cli_train_help():
    out = CliRunner().invoke(cli.train, ["-h"])
    assert out.exit_code == 0
    assert "Usage: train [OPTIONS]" in out.output
    assert "Train the model on a dataset" in out.output
    assert "Options:" in out.output
    assert "-m, --model TEXT           Path to save trained model to  [required]" in out.output
    assert "-d, --dataset TEXT         Path to dataset folder  [required]" in out.output
    assert "-c, --classes INTEGER      Number of classes (Default: 2)" in out.output
    assert "-e, --epochs INTEGER       Number of epochs (Default: 10)" in out.output
    assert "-b, --batch-size INTEGER   Batch size (Default: 100)" in out.output
    assert "-l, --learning-rate FLOAT  Learning rate (Default: 0.001)" in out.output
    assert "-L, --loss-goal FLOAT      If this loss is reached when training, it stops" in out.output
    assert "-C, --cpu                  Use CPU" in out.output
    assert "-w, --warnings             Show warnings" in out.output
    assert "-h, --help                 Show this message and exit." in out.output


def test_cli_eval():
    out = CliRunner().invoke(cli.eval)
    assert out.exit_code != 0
    assert "Usage: eval [OPTIONS]" in out.output
    assert "Try 'eval -h' for help." in out.output
    assert "Error: Missing option '-m' / '--model'." in out.output


def test_cli_eval_help():
    out = CliRunner().invoke(cli.eval, ["-h"])
    assert out.exit_code == 0
    assert "Usage: eval [OPTIONS]" in out.output
    assert "Evaluate an image using a trained model" in out.output
    assert "Options:" in out.output
    assert "-m, --model TEXT       Path to trained model  [required]" in out.output
    assert "-i, --image TEXT       Path to image  [required]" in out.output
    assert "-c, --classes INTEGER  Number of classes (Default: 2)" in out.output
    assert "-C, --cpu              Use CPU" in out.output
    assert "-w, --warnings         Show warnings" in out.output
    assert "-h, --help             Show this message and exit." in out.output
