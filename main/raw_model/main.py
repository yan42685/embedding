from raw_model.transE import TransE
from pathlib import Path
from raw_model.KG import KG


def main():
    kg_dir = Path.cwd().parent.joinpath("target/YG30K")
    model = TransE(kg_dir=kg_dir, epochs=40, batch_size=100, dimension=100, learning_rate=0.001, margin=4.0, norm="L1",
                   evaluation_mode="validation")
    model.train()


if __name__ == "__main__":
    main()
