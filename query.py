#-*- coding: UTF-8 -*-

from py2neo import Graph,Node,Relationship,NodeMatcher
from collections import defaultdict
class Query():
    def __init__(self):
        self.graph=Graph('http://localhost:7474/db/data/',name='stock',username='neo4j',password='neo4j')

    def run(self,cql):
        result = defaultdict(set)
        find_rela = self.graph.run(cql).to_ndarray()
        for i in find_rela:
            result[i[0]].add(i[1])
        return result
