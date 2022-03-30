import numpy as np
import codecs
import time
from pathlib import Path


def load_data(entity_file, relation_file, fact_file, is_quad=False, encoding="utf-8"):
    print("loading files...")

    entities = []
    relations = []
    facts = []

    with codecs.open(entity_file, "r", encoding=encoding) as file1, codecs.open(relation_file, "r",
                                                                                encoding=encoding) as file2, codecs.open(
        fact_file, "r", encoding=encoding) as file3:
        lines1 = file1.readlines()
        for line in lines1:
            line = line.strip().split("\t")
            entities.append(line[0])

        lines2 = file2.readlines()
        for line in lines2:
            line = line.strip().split("\t")
            relations.append(line[0])

        lines3 = file3.readlines()
        for line in lines3:
            fact = line.strip().split("\t")
            if is_quad:
                facts.append((fact[0], fact[1], fact[2]))
            else:
                facts.append(tuple(fact))

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


def exec_pipeline(input_param, *functions):
    """
    按顺序执行只有一个参数的函数序列，前一个函数的结果作为后一个函数的参数
    """
    counter = 0
    result = None
    for function in functions:
        counter += 1
        if counter == 1:
            result = function(input_param)
        else:
            result = function(result)
    return result


def read_lines(path, encoding="utf-8"):
    with codecs.open(path, "r", encoding=encoding) as file:
        result = file.readlines()
    return result


def mkdir(dir_path: Path):
    if not dir_path.is_dir():
        dir_path.mkdir()
