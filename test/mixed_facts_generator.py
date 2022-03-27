from pathlib import Path
from tools import time_it
import math
import codecs
import re
import collections
from annotated_facts_generator import AnnotatedFactsGenerator
import random


def main():
    generator = MixedFactsGenerator()
    generator.run()


class MixedFactsGenerator:
    def __init__(self):
        self.CHARSET = "utf-8"
        self.SEP = "\t"
        # 打算取的样本数
        self.SAMPLE_COUNT = 75000
        # 头实体出现次数下限, 用于筛选facts
        self.FILTER_THRESHOLD = 3
        self.input_path = Path.cwd().parent.joinpath("main/target/yago4-wd-facts.nt")
        self.output_path = Path.cwd().parent.joinpath("main/target/yago4_mixed_quads.txt")

    @time_it
    def run(self):
        quads = self._filter_quads(self.input_path)
        self.output_quads(quads)

    @time_it
    def _filter_quads(self, input_path):
        quads = []
        elements_pattern = re.compile(r'(?<=<)[^<>]+(?=>)')
        entity_dict = collections.defaultdict(int)
        entity_set = set()
        counter = 0
        with codecs.open(input_path, "r", encoding=self.CHARSET) as file:
            entity_set_max_count = math.ceil(self.SAMPLE_COUNT / self.FILTER_THRESHOLD)
            while len(entity_set) < entity_set_max_count:
                line = file.readline()
                elements = elements_pattern.findall(line)
                head_entity = elements[0]
                entity_dict[head_entity] += 1
                if entity_dict[head_entity] >= self.FILTER_THRESHOLD:
                    entity_set.add(head_entity)
                counter += 1

        with codecs.open(input_path, "r", encoding=self.CHARSET) as file:
            while len(quads) < self.SAMPLE_COUNT:
                line = file.readline()
                elements = elements_pattern.findall(line)
                if len(elements) == 3 and elements[0] in entity_set:
                    quads.append((elements[0], elements[1], elements[2], ""))

        return quads

    @time_it
    def output_quads(self, quads):
        annotated_quads_path = AnnotatedFactsGenerator.OUTPUT_PATH
        annotated_quads = []
        with codecs.open(annotated_quads_path, "r", encoding=self.CHARSET) as annotated_quads_file, \
                codecs.open(self.output_path, "w", encoding=self.CHARSET) as output_file:
            for line in annotated_quads_file.readlines():
                annotated_quads.append(tuple(line.split(self.SEP)))
            mixed_quads = annotated_quads + quads
            # random.shuffle(mixed_quads)
            print(len(mixed_quads))
            for quad in mixed_quads:
                new_line = self.SEP.join(quad)
                output_file.write(new_line)


if __name__ == "__main__":
    main()
