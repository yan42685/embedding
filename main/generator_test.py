from pathlib import Path
import pandas as pd

input_dir = Path.cwd()
facts_path = input_dir.joinpath("target/yagoFacts.tsv")
date_facts_path = input_dir.joinpath("target/yagoDateFacts.tsv")
meta_facts_path = input_dir.joinpath("target/yagoMetaFacts.tsv")

facts_df = pd.read_csv(facts_path, header=None, sep="\t")
date_facts_df = pd.read_csv(date_facts_path, header=None, sep="\t")
meta_facts_df = pd.read_csv(meta_facts_path, header=None, sep="\t")

facts_df.columns = ["fact_id", "head", "relation", "tail", "useless1"]
date_facts_df.columns = ["date_fact_id", "head", "relation", "useless2", "date"]
meta_facts_df.columns = ["meta_fact_id", "fact_id", "verb", "useless3", "date"]
# 合并facts和meta_facts
df1 = pd.merge(facts_df, meta_facts_df, on="fact_id")
df1 = df1.loc[df1["verb"] == "<occursSince>"]
df1 = df1.drop(columns=["meta_fact_id", "fact_id", "verb", "useless1", "useless3"])

# 合并facts和date_facts
# 数据集类型1: 大部分关系都有很强的时间顺序
# "<isAffiliatedTo>",
time_sensitive_relations = {"<wasBornIn>", "<hasWonPrize>", "<diedIn>", "<hasChild>",
                            "<graduatedFrom>", "<isMarriedTo>", "<worksAt>", "<directed>", "<isLeaderOf>",
                            "<isPoliticianOf>", "<actedIn>", "<wroteMusicFor>"}
#
df2 = facts_df.loc[facts_df["relation"].isin(time_sensitive_relations)]
df2 = pd.merge(df2, date_facts_df.drop(columns=["relation"]), on="head")
# [下面被注释的一行] 数据集类型2: 少部分关系有很强的时间顺序
# df2 = pd.merge(facts_df, date_facts_df.drop(columns=["relation"]), on="head")
df2 = df2.drop(columns=["date_fact_id", "fact_id", "useless1", "useless2"])

# 得到最终结果
df3 = pd.concat([df1, df2])
df3 = df3.dropna()
df3 = df3.drop_duplicates()

# =========== 测试 ============
h_threshold = 15
t_threshold = 8
head_counts = df3["head"].value_counts(sort=False)
df5 = df3[df3["head"].isin(head_counts.index[head_counts >= h_threshold])]


def condition1():
    return df5["head"].value_counts().tail(1).values[0] >= h_threshold


def condition2():
    return df5["tail"].value_counts().tail(1).values[0] >= t_threshold


while len(df5) > 0 and not (condition1() and condition2()):
    tail_counts = df5["tail"].value_counts(sort=False)
    df5 = df5[df5["tail"].isin(tail_counts.index[tail_counts >= t_threshold])]
    head_counts = df5["head"].value_counts(sort=False)
    df5 = df5[df5["head"].isin(head_counts.index[head_counts >= h_threshold])]
