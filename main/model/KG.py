import codecs
from pathlib import Path


class KG:
    _ENTITY_FILENAME = "entity.txt"
    _RELATION_FILENAME = "relation.txt"
    _TRAIN_FILENAME = "train.txt"
    _VALIDATION_FILENAME = "validation.txt"
    _TEST__FILENAME = "test.txt"

    def __init__(self, directory: Path, sep="\t"):
        self.directory = directory
        self.sep = sep
        self.entities = []
        self.relations = []
        self.train_quads = []
        self.validation_quads = []
        self.test_quads = []
        self.all_quads = []
        self._load_kg(directory)

    def _load_kg(self, directory):
        for line in self._read_lines(self._ENTITY_FILENAME):
            self.entities.append(line.strip("\t"))
        for line in self._read_lines(self._RELATION_FILENAME):
            self.relations.append(line.strip("\t"))
        for line in self._read_lines(self._TRAIN_FILENAME):
            self.train_quads.append(tuple(line.split(self.sep)))
        for line in self._read_lines(self._VALIDATION_FILENAME):
            self.validation_quads.append(tuple(line.split(self.sep)))
        for line in self._read_lines(self._TEST__FILENAME):
            self.test_quads.append(tuple(line.split(self.sep)))
        self.all_quads = self.train_quads + self.validation_quads + self.test_quads

    def _read_lines(self, filename):
        path = self.directory.joinpath(filename)
        with codecs.open(path, "r") as file:
            result = file.readlines()
        return result
