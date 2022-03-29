from tf_model.transE import TransE
import tensorflow as tf


def main():
    model = TransE()
    model.train()


if __name__ == '__main__':
    main()
