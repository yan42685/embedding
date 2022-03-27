from rdflib import Graph
from tools import time_it
import codecs
import pandas as pd
from pathlib import Path
import collections
import re


def main():
    generator = AnnotatedFactsGenerator()
    generator.run()


class AnnotatedFactsGenerator:
    def __init__(self):
        self.CHARSET = "utf-8"
        # 过滤出在头实体位置出现次数 >= threshold的实体所在在的四元组
        self.FILTER_THRESHOLD = 10
        self.input_path = Path.cwd().parent.joinpath("main/target/yago4-wd-annotated-facts.ntx")

        self.output_path = Path.cwd().parent.joinpath("main/target/yago4_annotated_quads.txt")

    @time_it
    def run(self):
        extracted_quads = self._extract_quads()
        filtered_quads = self._filter_quads(extracted_quads)
        self._output_quads(filtered_quads)

    @time_it
    def _extract_quads(self):
        with codecs.open(self.input_path, encoding=self.CHARSET) as file:
            text = file.readlines()
        elements_pattern = re.compile(r'(?<=<)[^<>]+(?=>)')
        date_pattern = re.compile(r'(?<=")\d{4}.*(?=")')
        quads = []
        initial_count = 0
        final_count = 0
        for line in text:
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

        print("filtered quad: %d" % len(entity_set))
        return filter(lambda quad: quad[0] in entity_set, quads)

    @time_it
    def _output_quads(self, quads):
        output_path = self.output_path
        with codecs.open(output_path, "w", encoding=self.CHARSET) as output_file:
            for quad in quads:
                new_line = "\t".join(quad)
                output_file.write(new_line + "\n")


if __name__ == "__main__":
    main()
