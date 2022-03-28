from tools import time_it, exec_pipeline
import codecs
from pathlib import Path

import collections
import re


def main():
    generator = AnnotatedFactsGenerator()
    generator.run()


class AnnotatedFactsGenerator:
    OUTPUT_PATH = Path.cwd().parent.joinpath("main/target/yago4_annotated_quads.txt")

    def __init__(self):
        self.CHARSET = "utf-8"
        self.SEP = "\t"
        # 过滤出在头实体位置出现次数 >= threshold的实体所在在的四元组
        self.FILTER_THRESHOLD = 15
        self.INPUT_PATH = Path.cwd().parent.joinpath("main/target/yago4-wd-annotated-facts.ntx")

    @time_it
    def run(self):
        exec_pipeline(self.INPUT_PATH, self._extract_quads, self._filter_quads, self._output_quads)

    @time_it
    def _extract_quads(self, input_path):
        with codecs.open(input_path, encoding=self.CHARSET) as file:
            lines = file.readlines()
        elements_pattern = re.compile(r'(?<=<)[^<>]+(?=>)')
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

        print("initial fact: %d, extracted quad: %d" % (initial_count, final_count))

        return quads

    @time_it
    def _filter_quads(self, quads):
        entity_dict = collections.defaultdict(int)
        entity_set = set()
        for (a, b, c, d) in quads:
            entity_dict[a] += 1
            entity_dict[c] += 1
            if entity_dict[a] >= self.FILTER_THRESHOLD:
                entity_set.add(a)
            if entity_dict[c] >= self.FILTER_THRESHOLD:
                entity_set.add(c)

        filtered_quads = list(filter(lambda quad: quad[0] in entity_set, quads))
        print("filtered quad: %d" % len(filtered_quads))
        return filtered_quads

    @time_it
    def _output_quads(self, quads):
        with codecs.open(self.OUTPUT_PATH, "w", encoding=self.CHARSET) as output_file:
            for quad in quads:
                new_line = self.SEP.join(quad)
                output_file.write(new_line + "\n")


if __name__ == "__main__":
    main()
