from typing import List, Dict
import numpy as np
from fglib import graphs, nodes, inference, rv
from fglib.nodes import VNode, FNode, NodeType
from fglib.edges import Edge
from fglib.rv import Gaussian
import networkx as nx
import random
from fglib.inference import loopy_belief_propagation



class DynaFNode(FNode):
    def __init__(self, label, factor=None, ):
        super().__init__(label, factor)
    def update_factor(self, x0: Gaussian, x1: Gaussian, dt: float = 1, precision: float = 1):
        # NOTE h(x) = v1 - v0, z = ?
        z = np.zeros((4, 1))
        v0 = x0.mean
        v1 = x1.mean
        v_ = np.vstack([v0, v1])  # [8, 1]

        # kinetic
        k = np.identity(4)   # [4, 4]
        k[:2, 2:] = np.identity(2) * dt

        h = k @ v0 - v1     # [4, 1]
        # jacob of h
        jacob = np.array([
            [1, 0, dt, 0, -1, 0, 0, 0],  # h(x)[0] = dx = x(k) + vx(k) * dt - x(k+1)
            [0, 1, 0, dt, 0, -1, 0, 0],  # h(x)[1] = dy = y(k) + vy(k) * dt - y(k+1)
            [0, 0, 1, 0, 0, 0, -1, 0],  # h(x)[2] = dvx = vx(k) - vx(k+1)
            [0, 0, 0, 1, 0, 0, 0, -1],  # h(x)[3] = dvy = vy(k) - vy(k+1)
        ])  # [4, 8]

        # presicion of observation z (i.e. the target of h) (here is zero)
        precision = np.array([
            [dt**3/3, 0, dt**2/2, 0],
            [0, dt**3/3, 0, dt**2/2],
            [dt**2/2, 0, dt, 0],
            [0, dt**2/2, 0, dt]
        ]).T * precision  # [4, 4]

        # NOTE https://arxiv.org/pdf/1910.14139.pdf
        w = jacob.T @ precision @ jacob
        wm = jacob.T @ precision @ (jacob @ v_ + z - h)

        self.factor = Gaussian.inf_form(w, wm, *x0.dim, *x1.dim)


class DistFNode1(FNode):
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
            if type(f) is DynaFNode:
                f.update_factor(vs[0].belief, vs[1].belief, precision=10)

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

    x0 = VNode("x0", Gaussian(np.random.rand(4, 1), np.diag([10] * 4), 'x0.x', 'x0.y', 'x0.vx', 'x0.vy'))
    x1 = VNode("x1", Gaussian(np.random.rand(4, 1), np.diag([10] * 4), 'x1.x', 'x1.y', 'x1.vx', 'x1.vy'))
    x2 = VNode("x2", Gaussian(np.random.rand(4, 1), np.diag([10] * 4), 'x2.x', 'x2.y', 'x2.vx', 'x2.vy'))
    x3 = VNode("x3", Gaussian(np.random.rand(4, 1), np.diag([10] * 4), 'x3.x', 'x3.y', 'x3.vx', 'x3.vy'))

    f0 = FNode("f0", Gaussian(np.array([0, 0, 2, 1])[..., None], np.diag([.01] * 4), 'x0.x', 'x0.y', 'x0.vx', 'x0.vy'))
    f01 = DynaFNode("f01", None)
    f12 = DynaFNode("f12", None)
    f23 = DynaFNode("f23", None)

    fg.set_nodes([x0, x1, x2, x3])
    fg.set_nodes([f0, f01, f12, f23])

    fg.set_edge(f0, x0)
    fg.set_edge(f01, x0)
    fg.set_edge(f01, x1)
    fg.set_edge(f12, x1)
    fg.set_edge(f12, x2)
    fg.set_edge(f23, x2)
    fg.set_edge(f23, x3)

    bs = loopy_belief_propagation(fg, 4)

    print('============')
    for v, p in bs.items():
        # print(v, p, p.dim)
        print(v, p.mean)

