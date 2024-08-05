


import ast
import csv
import os

import pandas as pd

import datasets



def _load_table_data(table_file):
    rows = []
    table_data = pd.read_csv(table_file)
    # the first line is header
    header = list(table_data.columns)
    for row_data in table_data.values:
        rows.append([str(_) for _ in list(row_data)])

    return header, rows


def _parse_answer_coordinates(answer_coordinate_str):

    try:
        answer_coordinates = []
        coords = ast.literal_eval(answer_coordinate_str)
        for row_index, column_index in sorted(ast.literal_eval(coord) for coord in coords):
            answer_coordinates.append({"row_index": row_index, "column_index": column_index})
        return answer_coordinates
    except SyntaxError:
        raise ValueError("Unable to evaluate %s" % answer_coordinate_str)


def _parse_answer_text(answer_text_str):
    try:
        answer_texts = []
        for value in ast.literal_eval(answer_text_str):
            answer_texts.append(value)
        return answer_texts
    except SyntaxError:
        raise ValueError("Unable to evaluate %s" % answer_text_str)


class MsrSQA(datasets.GeneratorBasedBuilder):


    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(

            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "annotator": datasets.Value("int32"),
                    "position": datasets.Value("int32"),
                    "question": datasets.Value("string"),
                    "question_and_history": datasets.Sequence(datasets.Value("string")),
                    "table_file": datasets.Value("string"),
                    "table_header": datasets.features.Sequence(datasets.Value("string")),
                    "table_data": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
                    "answer_coordinates": datasets.features.Sequence(
                        {"row_index": datasets.Value("int32"), "column_index": datasets.Value("int32")}
                    ),
                    "answer_text": datasets.features.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=None,

        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = os.path.join(dl_manager.download_and_extract(_URL), "SQA Release 1.0")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(data_dir, "random-split-1-train.tsv"), "data_dir": data_dir},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(data_dir, "random-split-1-dev.tsv"), "data_dir": data_dir},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(data_dir, "test.tsv"), "data_dir": data_dir},
            ),
        ]

    def _generate_examples(self, filepath, data_dir):

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            question_and_history = []
            for idx, item in enumerate(reader):
                item["answer_text"] = _parse_answer_text(item["answer_text"])
                item["answer_coordinates"] = _parse_answer_coordinates(item["answer_coordinates"])
                header, table_data = _load_table_data(os.path.join(data_dir, item["table_file"]))
                item["table_header"] = header
                item["table_data"] = table_data
                if item["position"] == "0":
                    question_and_history = []  # reset history
                question_and_history.append(item["question"])
                item["question_and_history"] = question_and_history
                yield idx, item