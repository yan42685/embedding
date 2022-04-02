from raw_model.time_transE import TimeTransE
from raw_model.transE import TransE
from raw_model.transH import TransH
from pathlib import Path
from raw_model.KG import KG


def main():
    kg_dir = Path.cwd().parent.joinpath("target/YG30K")
    # model = TransE(kg_dir=kg_dir, epochs=10, batch_size=100, dimension=100, learning_rate=0.001, margin=4.0, norm="L1",
    #                epsilon=0.9,
    #                evaluation_mode="validation")

    # model = TransE(kg_dir=Path.cwd().parent.joinpath("target/YG15K"))

    # model = TransH(kg_dir=kg_dir)

    model = TimeTransE(kg_dir=kg_dir)
    model.train()


if __name__ == "__main__":
    main()
