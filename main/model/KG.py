import codecs


class KG:

    def __init__(self, file_path, sep="\t"):
        self.sep = sep
        self.entities = []
        self.relations = []
        self.facts = []
        self._load_kg(file_path)

    def _load_kg(self, file_path):
        entity_set = set()
        relation_set = set()
        facts = []
        with codecs.open(file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                quad = tuple(line.split(self.sep))
                entity_set.add(quad[0])
                entity_set.add(quad[2])
                relation_set.add(quad[1])
                facts.append(quad)

        self.entities = list(entity_set)
        self.relations = list(relation_set)
        self.facts = facts


