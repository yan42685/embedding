from pathlib import Path
from rdflib import Graph


class MixedFactsGenerator:
    def __init__(self):
        self.input_path = Path.cwd().parent.joinpath("main/target/yago4-wd-facts.ntx")
        self.output_path = Path.cwd().parent.joinpath("main/target/yago4_mixed_quads.txt")
