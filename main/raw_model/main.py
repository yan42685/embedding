from raw_model.transE import TransE
from tools import load_data


def main(data_set_name):
    if data_set_name == "free_base":
        entity_file = "../data_set/FB15k/entity2id.txt"
        relation_file = "../data_set/FB15k/relation2id.txt"
        fact_file = "../data_set/FB15k/test.txt"
    elif data_set_name == "word_net":
        entity_file = "../data_set/WN18/entity2id.txt"
        relation_file = "../data_set/WN18/relation2id.txt"
        fact_file = "../data_set/WN18/wordnet-mlj12-test.txt"
    else:
        raise RuntimeError("Wrong data set name")
    entities, relations, facts = load_data(entity_file, relation_file, fact_file)

    model = TransE(entities, relations, facts, dimension=50, learning_rate=0.01, margin=1.0, norm=1)
    model.train(epoch_count=3, data_set_name=data_set_name)


if __name__ == "__main__":
    main("word_net")
