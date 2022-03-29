from model.transE import TransE
import tensorflow as tf


def main():
    # 避免一些版本2的坑
    model = TransE()
    model.train()


if __name__ == '__main__':
    main()
