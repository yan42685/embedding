from tools import time_it, exec_pipeline
import codecs
from pathlib import Path

import collections
import re


def main():
    # threshold = 14 对应99k quads     =15 对应75k quads
    generator = AnnotatedDataGenerator(filter_threshold=14)
    generator.run()


class AnnotatedDataGenerator:
    OUTPUT_ENTITY_PATH = Path.cwd().parent.joinpath("main/target/yago4_annotated_entities.txt")
    OUTPUT_RELATION_PATH = Path.cwd().parent.joinpath("main/target/yago4_annotated_relations.txt")
    OUTPUT_QUAD_PATH = Path.cwd().parent.joinpath("main/target/yago4_annotated_quads.txt")

    def __init__(self, filter_threshold=14):
        self.CHARSET = "utf-8"
        self.SEP = "\t"
        # 过滤出在头实体位置出现次数 >= threshold的实体所在在的四元组
        self.filter_threshold = filter_threshold
        self.INPUT_PATH = Path.cwd().parent.joinpath("main/target/yago4-wd-annotated-facts.ntx")

    @time_it
    def run(self):
        exec_pipeline(self.INPUT_PATH, self._extract_quads, self._filter_data, self._output_quads)

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
    def _filter_data(self, quads):
        entity_dict = collections.defaultdict(int)
        entity_set = set()
        for (a, b, c, d) in quads:
            entity_dict[a] += 1
            entity_dict[c] += 1
            if entity_dict[a] >= self.filter_threshold:
                entity_set.add(a)
            if entity_dict[c] >= self.filter_threshold:
                entity_set.add(c)

        filtered_quads = list(filter(lambda quad: quad[0] in entity_set, quads))
        print("filtered quad: %d" % len(filtered_quads))

        filtered_entities = set()
        filtered_relations = set()
        for (a, b, c, d) in filtered_quads:
            filtered_entities.add(a)
            filtered_entities.add(c)
            filtered_relations.add(b)
        print("filtered entities: %d" % len(filtered_entities))
        print("filtered relations: %d" % len(filtered_relations))
        return filtered_entities, filtered_relations, filtered_quads

    @time_it
    def _output_quads(self, data):
        entities = data[0]
        relations = data[1]
        quads = data[2]
        with codecs.open(self.OUTPUT_ENTITY_PATH, "w", encoding=self.CHARSET) as entity_file:
            for entity in entities:
                entity_file.write(entity + "\n")
        with codecs.open(self.OUTPUT_RELATION_PATH, "w", encoding=self.CHARSET) as relation_file:
            for relation in relations:
                relation_file.write(relation + "\n")
        with codecs.open(self.OUTPUT_QUAD_PATH, "w", encoding=self.CHARSET) as quad_file:
            for quad in quads:
                new_line = self.SEP.join(quad)
                quad_file.write(new_line + "\n")


if __name__ == "__main__":
    main()
