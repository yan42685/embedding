from pathlib import Path
import pandas as pd


def main():
    test_input_dir = Path.cwd().joinpath("target")
    TestGenerator(input_dir=test_input_dir).generate()


class TestGenerator:
    def __init__(self, input_dir, threshold=40, sep="\t"):
        self.input_dir = input_dir
        self.facts_path = input_dir.joinpath("yago3Facts.tsv")
        self.date_facts_path = input_dir.joinpath("yago3DateFacts.tsv")
        self.meta_facts_path = input_dir.joinpath("yago3MetaFacts.tsv")
        self.threshold = threshold
        self.sep = sep

        self.annotated_output_dir = input_dir.joinpath("YG15")
        self.mixed_output_dir = input_dir.joinpath("YG30K")
        self.entities = set()
        self.relations = set()
        self.all_quads = []
        self.annotated_quads = []
        self.mixed_quads = []


    def generate(self):
        print("Start generating...")
        self._extract_quads()
        pass

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
        return list(df4.itertuples(index=False, name=None))


if __name__ == "__main__":
    main()
