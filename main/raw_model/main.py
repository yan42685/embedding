from raw_model.time_transE import TimeTransE
from raw_model.transE import TransE
from raw_model.transH import TransH
from pathlib import Path
from raw_model.KG import KG


def main():
    kg_30K_dir = Path.cwd().parent.joinpath("target/YG30K")
    kg_15K_dir = Path.cwd().parent.joinpath("target/YG15K")
    # model = TransE(kg_dir=kg_30K_dir, epochs=20, batch_size=100, dimension=100, learning_rate=0.001, margin=4.0,
    #                norm="L1",
    #                epsilon=0.9,
    #                evaluation_mode="validation")
    #
    # model = TransE(kg_dir=kg_15K_dir, epochs=20)

    model = TimeTransE(kg_dir=kg_30K_dir, epochs=40, batch_size=100, dimension=100, learning_rate=0.001, margin=4.0,
                       norm="L1",
                       epsilon=0.9,
                       evaluation_mode="validation", k=0.01)

    # model = TimeTransE(kg_dir=kg_15K_dir, epochs=20)

    # TODO: 解决evaluator的问题, self.relation_embedding为空
    # model = TransH(kg_dir=Path.cwd().parent.joinpath("target/YG15K"))

    model.train()


if __name__ == "__main__":
    main()
