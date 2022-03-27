from rdflib import Graph
from tools import time_it
import codecs
import pandas as pd
from pathlib import Path
import re

yago4_annotated_facts_path = Path.cwd().parent.joinpath("main/target/yago4-wd-annotated-facts.ntx")
test_annotated_facts_path = Path.cwd().parent.joinpath("main/target/test_annotated_facts.ntx")


def main():
    # get_time_annotated_facts()
    # test_filter()
    # sample_text()
    test_regex()


@time_it
def read_ntx(path):
    graph = Graph()
    # 约600万个，读取需要约400秒
    graph.parse(location=path, format="nt")
    print("triple count: %d" % len(graph))
    return graph


@time_it
def test_filter():
    facts = read_ntx(test_annotated_facts_path)

    list1 = []

    for triple in facts:
        list1.append(triple)

    data_frame = pd.DataFrame(list1)
    result1 = data_frame.apply(pd.value_counts)
    print(len(result1))
    print(result1.head())

    result1.columns = ["h_count", "r_count", "t_count"]
    result2 = result1.loc[result1.h_count + result1.t_count >= 2]

    print(len(result2))
    print(result2.head())


@time_it
def get_time_annotated_facts():
    meta_facts = read_ntx(test_annotated_facts_path)
    for i in meta_facts:
        print(i)


def sample_text():
    with codecs.open(yago4_annotated_facts_path, "r", encoding="utf-8") as source_file, \
            codecs.open(test_annotated_facts_path, "w", encoding="utf-8") as target_file:
        for i in range(20000):
            line = source_file.readline()
            target_file.write(line)

    read_ntx(test_annotated_facts_path)


def test_regex():
    element_pattern = re.compile(r'(?<=<)[^<>]+(?=>)')
    date_pattern = re.compile(r'(?<=")\d{4}.*(?=")')
    with codecs.open(test_annotated_facts_path, "r", encoding="utf-8") as file:
        for i in range(100):
            line = file.readline()
            elements = element_pattern.findall(line)
            print(elements)
            date = date_pattern.findall(line)
            print(date)


if __name__ == "__main__":
    main()
