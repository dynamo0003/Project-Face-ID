import sys

sys.path.append("src")

import cli


def test_parse_args():
    argv = sys.argv.copy()
    sys.argv = [
        "",
        "model",
        "-i",
        "img",
        "-e",
        "1",
        "-b",
        "2",
        "-l",
        "3",
        "-c",
        "-w",
    ]
    args = cli.parse_args()
    assert args.model == "model"
    assert args.image == "img"
    assert args.epochs == 1
    assert args.batch_size == 2
    assert args.learning_rate == 3
    assert args.cpu
    assert args.warnings
    sys.argv = argv
