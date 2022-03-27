import numpy as np
import codecs
import time
from os.path import dirname, abspath


def load_data(entity_file, relation_file, fact_file):
    print("loading files...")

    entities = []
    relations = []
    facts = []

    with codecs.open(entity_file, "r") as file1, codecs.open(relation_file, "r") as file2, codecs.open(fact_file,
                                                                                                       "r") as file3:
        lines1 = file1.readlines()
        for line in lines1:
            line = line.strip().split("\t")
            if len(line) != 2:
                continue
            entities.append(line[0])

        lines2 = file2.readlines()
        for line in lines2:
            line = line.strip().split("\t")
            if len(line) != 2:
                continue
            relations.append(line[0])

        lines3 = file3.readlines()
        for line in lines3:
            fact = line.strip().split("\t")
            if len(fact) != 3:
                continue
            facts.append(fact)

    print("Loading complete. entity: %d , relation: %d , fact: %d" % (
        len(entities), len(relations), len(facts)))

    return entities, relations, facts


def generate_initial_vector(dimension):
    return np.random.uniform(-6.0 / np.sqrt(dimension), 6.0 / np.sqrt(dimension),
                             dimension)


# 曼哈顿距离
def norm_l1(vector):
    return np.linalg.norm(vector, ord=1)


# 欧氏距离
def norm_l2(vector):
    return np.linalg.norm(vector, ord=2)


# 缩放到欧氏距离的单位长度
def scale_to_unit_length(vector):
    return vector / norm_l2(vector)


# 统计代码耗时的装饰器
def time_it(func):
    def func_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f'function [{func.__name__}] costs time: {end_time - start_time:.6f}s')
        return result

    return func_wrapper


