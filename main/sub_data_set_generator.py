from tools import time_it, mkdir
from sklearn.model_selection import train_test_split
import codecs
import pandas as pd
from pathlib import Path
import random

import collections
import re


# 生成子数据集

def main():
    test_input_path = Path.cwd().joinpath("target/yago4-wd-annotated-facts.ntx")
    generator = SubDataSetGenerator(input_path=test_input_path)
    generator.run()


class SubDataSetGenerator:
    _DEFAULT_INPUT_PATH = Path.cwd().joinpath("data_set").joinpath("yago4-wd-annotated-facts.ntx")

    def __init__(self, filter_threshold=10, max_quads_count=30000, input_path=_DEFAULT_INPUT_PATH, charset="utf-8",
                 sep="\t"):
        self.CHARSET = charset
        self.SEP = sep
        self.filter_threshold = filter_threshold
        self.max_quads_count = max_quads_count
        self.input_path = input_path
        self.annotated_output_dir = input_path.parent.joinpath("YG15K")
        self.mixed_output_dir = input_path.parent.joinpath("YG30K")
        self.entities = set()
        self.relations = set()
        self.all_quads = []
        self.annotated_quads = []
        self.mixed_quads = []

    @time_it
    def run(self):
        extract_quads = self._extract_quads()
        self._filter_quads(extract_quads)
        self._output_data(self.annotated_output_dir, self.annotated_quads)
        self._output_data(self.mixed_output_dir, self.mixed_quads)

    @time_it
    def _extract_quads(self):
        with codecs.open(self.input_path, encoding=self.CHARSET) as file:
            lines = file.readlines()
        # 匹配 <abc> 内的字符串
        elements_pattern = re.compile(r'(?<=<)[^<>]+(?=>)')
        # 匹配"1999"或"1999-01-01"这样的字符串双引号里的内容
        date_pattern = re.compile(r'(?<=")\d{4}.*(?=")')
        quads = []
        initial_count = 0
        final_count = 0
        for line in lines:
            initial_count += 1
            elements = elements_pattern.findall(line)
            if len(elements) == 5 and elements[3].endswith("startDate"):
                final_count += 1
                date = date_pattern.findall(line)[0]
                quad = (elements[0], elements[1], elements[2], date)
                quads.append(quad)

        print("initial facts: %d, extracted quads: %d" % (initial_count, final_count))

        return quads

    @time_it
    def _filter_quads(self, quads):
        entity_dict = collections.defaultdict(list)
        # 过滤出在头实体位置出现次数 在[threshold, 2 * threshold] 的实体所在在的四元组
        for (h, r, t, d) in quads:
            entity_dict[h].append((h, r, t, d))

        for quads in entity_dict.values():
            if self.filter_threshold <= len(quads) <= 2 * self.filter_threshold:
                self.all_quads.extend(quads)
                if len(self.all_quads) > self.max_quads_count:
                    break

        print("filtered quads: %d" % len(self.all_quads))

        for (h, r, t, d) in self.all_quads:
            self.entities.add(h)
            self.entities.add(t)
            self.relations.add(r)
        print("filtered entities: %d" % len(self.entities))
        print("filtered relations: %d" % len(self.relations))

        # 得到annotated_quads
        sample_count = int(len(self.all_quads) / 2)
        sampled_quads = random.sample(self.all_quads, sample_count)
        sampled_entities = set()
        sampled_relations = set()
        for (h, r, t, d) in sampled_quads:
            sampled_entities.add(h)
            sampled_entities.add(t)
            sampled_relations.add(r)
        print("sampled entities: %d, total entities: %d" % (len(sampled_entities), len(self.entities)))
        print("sampled relations: %d, total relations: %d" % (len(sampled_relations), len(self.relations)))
        self.annotated_quads = sampled_quads

        # 得到mixed_quads
        self.mixed_quads = list(self.annotated_quads)
        difference_set = set(self.all_quads).difference(self.annotated_quads)
        for (h, r, t, d) in difference_set:
            self.mixed_quads.append((h, r, t, "None"))

        print("annotated quads: %d,   mixed quads:%d" % (len(self.annotated_quads), len(self.mixed_quads)))
        random.shuffle(self.mixed_quads)

    @time_it
    def _output_data(self, directory, quads):
        entity_filename = "entity.txt"
        relation_filename = "relation.txt"
        train_filename = "train.txt"
        validation_filename = "validation.txt"
        test_filename = "test.txt"

        mkdir(directory)
        train_quads, validation_quads, test_quads = self._split_quads(quads)
        self._pandas_write(directory.joinpath(entity_filename), self.entities)
        self._pandas_write(directory.joinpath(relation_filename), self.relations)
        self._pandas_write(directory.joinpath(train_filename), train_quads)
        self._pandas_write(directory.joinpath(validation_filename), validation_quads)
        self._pandas_write(directory.joinpath(test_filename), test_quads)

    @staticmethod
    def _split_quads(quads):
        data_set = pd.DataFrame(quads)
        # 按 8:1:1 划分数据集
        train_quads, tmp_set = train_test_split(data_set, train_size=0.8, random_state=123)
        validation_quads, test_quads = train_test_split(tmp_set, train_size=0.5, random_state=123)
        print("train quads: %d, validation quads: %d, test quads: %d" % (
            len(train_quads), len(validation_quads), len(test_quads)))
        return train_quads, validation_quads, test_quads

    def _pandas_write(self, path, data):
        df = pd.DataFrame(data)
        df.to_csv(path, sep=self.SEP, index=False, header=None)


if __name__ == "__main__":
    main()
