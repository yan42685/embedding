from raw_model.transE import TransE
from pathlib import Path
from raw_model.KG import KG


def main():
    directory = Path.cwd().parent.joinpath("target/YG30K")
    kg = KG(directory=directory)
    model = TransE(kg=kg, epochs=1, batch_size=50, dimension=50, learning_rate=0.001, margin=4.0, norm="L1",
                   evaluation_mode="validation")
    model.train()


if __name__ == "__main__":
    main()
