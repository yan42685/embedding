from rdflib import Graph
from tools import time_it


@time_it
def read_ttl():
    graph = Graph()
    # 约600万个，读取需要约400秒
    # path = "main/target/yagoFacts.ttl"
    # 3600个，读取需要0.25秒
    test_facts_path = "main/target/testFacts.ttl"
    test_date_facts_path = "main/target/testDateFacts.ttl"
    test_meta_facts_path = "main/target/testMetaFacts.ttl"
    graph.parse(location=test_meta_facts_path, format="ttl")
    print(len(graph))
    count = 0
    for (a, b, c) in graph:
        print((a, b, c))
        print("type of a: %s    type of b: %s    type of c: %s" % (type(a), type(b), type(c)))
        print((len(a), len(b), len(c)))
        count += 1
        print(count)
        if count == 50:
            break


read_ttl()
