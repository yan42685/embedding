from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from tools import time_it, mkdir
import random


def main():
    test_input_dir = Path.cwd().joinpath("target")
    SubDataSetGenerator(input_dir=test_input_dir).generate()


class SubDataSetGenerator:
    def __init__(self, input_dir, threshold=40, sep="\t"):
        self.input_dir = input_dir
        self.facts_path = input_dir.joinpath("yagoFacts.tsv")
        self.date_facts_path = input_dir.joinpath("yagoDateFacts.tsv")
        self.meta_facts_path = input_dir.joinpath("yagoMetaFacts.tsv")
        self.threshold = threshold
        self.sep = sep

        self.annotated_output_dir = input_dir.joinpath("YG15K")
        self.mixed_output_dir = input_dir.joinpath("YG30K")
        self.entities = set()
        self.relations = set()
        self.all_quads = []
        self.annotated_quads = []
        self.mixed_quads = []

    @time_it
    def generate(self):
        print("Start generating...")
        self._extract_quads()
        self._filter_quads()
        self._output_data(self.annotated_output_dir, self.annotated_quads)
        self._output_data(self.mixed_output_dir, self.mixed_quads)
        print("Generating complete")


    def _extract_quads(self):
        facts_df = pd.read_csv(self.facts_path, header=None, sep=self.sep)
        date_facts_df = pd.read_csv(self.date_facts_path, header=None, sep=self.sep)
        meta_facts_df = pd.read_csv(self.meta_facts_path, header=None, sep=self.sep)

        facts_df.columns = ["fact_id", "head", "relation", "tail", "useless1"]
        date_facts_df.columns = ["date_fact_id", "head", "relation", "useless2", "date"]
        meta_facts_df.columns = ["meta_fact_id", "fact_id", "verb", "useless3", "date"]
        # 合并facts和meta_facts
        df1 = pd.merge(facts_df, meta_facts_df, on="fact_id")
        df1 = df1.loc[df1["verb"] == "<occursSince>"]
        df1 = df1.drop(columns=["meta_fact_id", "fact_id", "verb", "useless1", "useless3"])

        # 合并facts和date_facts
        # 数据集类型1: 大部分关系都有很强的时间顺序
        time_sensitive_relations = {"<wasBornIn>", "<isAffiliatedTo>", "<hasWonPrize>", "<diedIn>", "<hasChild>",
                                    "<graduatedFrom>", "<isMarriedTo>", "<worksAt>", "<directed>", "<isLeaderOf>"}

        df2 = facts_df.loc[facts_df["relation"].isin(time_sensitive_relations)]
        df2 = pd.merge(df2, date_facts_df.drop(columns=["relation"]), on="head")
        # 数据集类型2: 少部分关系有很强的时间顺序 threshold 取200
        # df2 = pd.merge(facts_df, date_facts_df.drop(columns=["relation"]), on="head")
        df2 = df2.drop(columns=["date_fact_id", "fact_id", "useless1", "useless2"])

        # 得到最终结果
        df3 = pd.concat([df1, df2])
        df3 = df3.dropna()
        df3 = df3.drop_duplicates()

        counts = df3['head'].value_counts(sort=False)
        # 头实体重复出现次数大于等于threshold的quad
        df4 = df3[df3['head'].isin(counts.index[counts >= self.threshold])]
        self.all_quads = list(df4.itertuples(index=False, name=None))

    def _filter_quads(self):
        for (h, r, t, d) in self.all_quads:
            self.entities.add(h)
            self.relations.add(r)
            self.entities.add(t)

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
            self.mixed_quads.append((h, r, t, -1))

        print("annotated quads: %d,   mixed quads:%d" % (len(self.annotated_quads), len(self.mixed_quads)))
        random.shuffle(self.annotated_quads)
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
        train_quads, tmp_set = train_test_split(data_set, train_size=0.85, random_state=123)
        validation_quads, test_quads = train_test_split(tmp_set, train_size=0.5, random_state=123)
        print("train quads: %d, validation quads: %d, test quads: %d" % (
            len(train_quads), len(validation_quads), len(test_quads)))
        return train_quads, validation_quads, test_quads

    def _pandas_write(self, path, data):
        df = pd.DataFrame(data)
        df.to_csv(path, sep=self.sep, index=False, header=None)


if __name__ == "__main__":
    main()
