from rdflib import Graph
from tools import time_it
import codecs
import pandas as pd
from pathlib import Path
import re


def main():
    generator = TestSubDataSetGenerator()
    generator.run()


class TestSubDataSetGenerator:
    def __init__(self):
        self.charset = "utf-8"
        self.yago4_annotated_facts_path = Path.cwd().parent.joinpath("main/target/yago4-wd-annotated-facts.ntx")
        self.test_annotated_facts_path = Path.cwd().parent.joinpath("main/target/test_annotated_facts.ntx")
        self.test2_annotated_facts_path = Path.cwd().parent.joinpath("main/target/test2_annotated_facts.txt")

    def run(self):
        sample_text = self._sample_text(20000)
        quads = self._filter_quads(sample_text)
        entities = self._filter_entities(quads)

    @time_it
    def _sample_text(self, lines):
        sample_text = []
        with codecs.open(self.yago4_annotated_facts_path, "r", encoding=self.charset) as source_file:
            for i in range(lines):
                sample_text.append(source_file.readline())
        return sample_text

    @time_it
    def _filter_quads(self, text):
        element_pattern = re.compile(r'(?<=<)[^<>]+(?=>)')
        date_pattern = re.compile(r'(?<=")\d{4}.*(?=")')
        quads = []
        initial_count = 0
        final_count = 0
        for line in text:
            initial_count += 1
            elements = element_pattern.findall(line)
            if len(elements) == 5 and elements[3].endswith("startDate"):
                final_count += 1
                date = date_pattern.findall(line)[0]
                quad = (elements[0], elements[1], elements[2], date)
                quads.append(quad)

        print("origin count: %d, final count: %d" % (initial_count, final_count))
        return quads

    @time_it
    def _filter_entities(self, quads):
        data_frame = pd.DataFrame(quads)
        result1 = data_frame.apply(pd.value_counts)
        print("total entities: %d" % len(result1))
        print(result1)

        result1.columns = ["h_count", "r_count", "t_count", "date"]
        result2 = result1.loc[result1.h_count >= 2]

        print("valid entities: %d" % len(result2))
        print(result2)


if __name__ == "__main__":
    main()
