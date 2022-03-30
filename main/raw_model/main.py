from raw_model.transE import TransE
from tools import load_data


def main(data_set_name):
    is_quad = False
    if data_set_name == "free_base":
        entity_file = "../data_set/FB15k/entity2id.txt"
        relation_file = "../data_set/FB15k/relation2id.txt"
        fact_file = "../data_set/FB15k/test.txt"
    elif data_set_name == "word_net":
        entity_file = "../data_set/WN18/entity2id.txt"
        relation_file = "../data_set/WN18/relation2id.txt"
        fact_file = "../data_set/WN18/wordnet-mlj12-test.txt"
    elif data_set_name == "yago4":
        entity_file = "../target/YG15K/entity.txt"
        relation_file = "../target/YG15K/relation.txt"
        fact_file = "../target/YG15K/train.txt"
        is_quad = True
    else:
        raise RuntimeError("Wrong data set name")
    entities, relations, facts = load_data(entity_file, relation_file, fact_file, is_quad=is_quad)

    # margin = 1的效果比 = 2 的效果好
    model = TransE(entities, relations, facts, dimension=50, learning_rate=0.01, margin=2.0, norm=1)
    model.train(epoch_count=3, data_set_name=data_set_name)


if __name__ == "__main__":
    # main("word_net")
    main("yago4")
