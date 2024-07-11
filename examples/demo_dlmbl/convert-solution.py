import argparse

from nbconvert.exporters import NotebookExporter
from nbconvert.preprocessors import ClearOutputPreprocessor, TagRemovePreprocessor
from traitlets.config import Config


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file")
    parser.add_argument("output_file")

    return parser


def convert(input_file, output_file):
    c = Config()
    c.TagRemovePreprocessor.remove_cell_tags = ("solution",)
    c.TagRemovePreprocessor.enabled = True
    c.ClearOutputPreprocesser.enabled = True
    c.NotebookExporter.preprocessors = [
        "nbconvert.preprocessors.TagRemovePreprocessor",
        "nbconvert.preprocessors.ClearOutputPreprocessor",
    ]

    exporter = NotebookExporter(config=c)
    exporter.register_preprocessor(TagRemovePreprocessor(config=c), True)
    exporter.register_preprocessor(ClearOutputPreprocessor(), True)

    output = NotebookExporter(config=c).from_filename(input_file)
    with open(output_file, "w") as f:
        f.write(output[0])


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    convert(args.input_file, args.output_file)
    print(f"Converted {args.input_file} to {args.output_file}")
