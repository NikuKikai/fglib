from typing import List, Dict
import numpy as np
from fglib import graphs, nodes, inference, rv
from fglib.nodes import VNode, FNode, NodeType
from fglib.edges import Edge
from fglib.rv import Gaussian
import networkx as nx
import random
from fglib.inference import loopy_belief_propagation


class DistFNode(FNode):
    def __init__(self, label, factor=None, ):
        super().__init__(label, factor)
    def update_factor(self, p_v0: Gaussian, p_v1: Gaussian, precision: float = 1):
        # NOTE h(x) = v1 - v0, z = ?
        z = np.array([[1], [2]])
        v0 = p_v0.mean
        v1 = p_v1.mean
        v_ = np.vstack([v0, v1])  # [4, 1]
        jacob = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]])  # h(x) is linear, [2, 4]

        h = v1 - v0     # [2, 1]
        jm = jacob @ v_ + z - h  # [2, 1]

        # presicion of observation z (i.e. the target of h) (here is zero)
        precision = np.array([[precision, 0], [0, precision]])
        w = jacob.T @ precision @ jacob
        wm = jacob.T @ precision @ jm

        self.factor = Gaussian.inf_form(w, wm, *p_v0.dim, *p_v1.dim)



def loopy_belief_propagation(graph: graphs.FactorGraph, iters: int = 4) -> Dict[VNode, rv.RandomVariable]:
    """Loopy Belief propagation. Returns each variable's final belief.
    """

    vnodes: List[VNode] = graph.get_vnodes()
    fnodes: List[FNode] = graph.get_fnodes()

    # Get edges in v -> n order
    edges = nx.edges(graph)
    edges = [(n0, n1) if n0.type==NodeType.variable_node else (n1, n0) for n0, n1 in edges]

    # Var Belief
    beliefs = {}

    for iter in range(iters):
        # Var -> Factor
        for v, f in edges:
            msg = v.spa(f)
            e: Edge = graph[v][f]['object']
            e.set_message(v, f, msg)

        # Factor Update
        for f in fnodes:
            vs: List[VNode] = list(graph.neighbors(f))
            if type(f) is DistFNode:
                f.update_factor(vs[0].belief, vs[1].belief)

        # Factor -> Var
        for v, f in edges:
            msg = f.spa(v)
            e: Edge = graph[f][v]['object']
            e.set_message(f, v, msg)

        # Var Belief Update
        for v in vnodes:
            beliefs[v] = v.update_belief()
    return beliefs


if __name__ == '__main__':
    fg = graphs.FactorGraph()

    x1 = VNode("x1", Gaussian([[0], [0]], [[10, 0], [0, 10]], 'x1.x', 'x1.y'))
    x2 = VNode("x2", Gaussian([[2], [2]], [[10, 0], [0, 10]], 'x2.x', 'x2.y'))
    x3 = VNode("x3", Gaussian([[0], [1]], [[10, 0], [0, 10]], 'x3.x', 'x3.y'))

    f1 = FNode("f1", Gaussian([[1], [1]], [[1, 0], [0, 1]], 'x1.x', 'x1.y'))
    f12 = DistFNode("f12", None)
    f23 = DistFNode("f23", None)
    f13 = DistFNode("f13", None)

    fg.set_nodes([x1, x2, x3])
    fg.set_nodes([f1, f12, f23, f13])

    fg.set_edge(f1, x1)
    fg.set_edge(f12, x1)
    fg.set_edge(f12, x2)
    fg.set_edge(f23, x2)
    fg.set_edge(f23, x3)
    fg.set_edge(f13, x1)
    fg.set_edge(f13, x3)

    bs = loopy_belief_propagation(fg, 4)

    print('============')
    for v, p in bs.items():
        # print(v, p, p.dim)
        print(v, p.mean)

