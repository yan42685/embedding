from pathlib import Path
from tools import time_it
import pandas as pd
import numpy as np


class KG:
    _DEFAULT_DIR = Path.cwd().parent.joinpath("target/YG30K")

    _ENTITY_FILENAME = "entity.txt"
    _RELATION_FILENAME = "relation.txt"
    _TRAIN_FILENAME = "train.txt"
    _VALIDATION_FILENAME = "validation.txt"
    _TEST__FILENAME = "test.txt"

    def __init__(self, directory: Path = _DEFAULT_DIR, sep="\t"):
        self.directory = directory
        self.sep = sep
        self.entity_id_dict = {}
        self.entity_ids = []
        self.relation_id_dict = {}
        self.relation_ids = []
        self.all_quads = []
        self.train_quads = []
        self.train_quads_set = set()
        self.validation_quads = []
        self.test_quads = []
        self._load_kg()

    @time_it
    def _load_kg(self):
        entities = self._pd_read(self._ENTITY_FILENAME)
        self.entity_ids = list(np.arange(len(entities[0])))
        self.entity_id_dict = dict(zip(entities[0], self.entity_ids))
        relations = self._pd_read(self._RELATION_FILENAME)
        self.relation_ids = list(np.arange(len(relations[0])))
        self.relation_id_dict = dict(zip(relations[0], self.relation_ids))

        train_quads = self._pd_read(self._TRAIN_FILENAME)
        self.train_quads = list(train_quads.apply(self._quad2ids, axis=1))
        self.train_quads_set = set(self.train_quads)
        validation_quads = self._pd_read(self._VALIDATION_FILENAME)
        self.validation_quads = list(validation_quads.apply(self._quad2ids, axis=1))
        test_quads = self._pd_read(self._TEST__FILENAME)
        self.test_quads = list(test_quads.apply(self._quad2ids, axis=1))
        self.all_quads = self.train_quads + self.validation_quads + self.test_quads

        print("knowledge graph loading complete.")
        print("entities: %d, relations: %d" % (len(self.entity_id_dict.values()), len(self.relation_id_dict.values())))
        print("train quads: %d, validation quads: %d, test quads: %d" % (
            len(self.train_quads), len(self.validation_quads), len(self.test_quads)), )

    def _pd_read(self, filename):
        path = self.directory.joinpath(filename)
        return pd.read_table(path, header=None, sep=self.sep)

    def _quad2ids(self, quad):
        head_id = self.entity_id_dict[quad[0]]
        relation_id = self.relation_id_dict[quad[1]]
        tail_id = self.entity_id_dict[quad[2]]
        date = quad[3]
        return head_id, relation_id, tail_id, date
