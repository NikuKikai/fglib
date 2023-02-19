from typing import List, Dict
from fglib.graphs import FactorGraph
from fglib.nodes import VNode, FNode
from fglib.rv import Gaussian
from fglib.inference import loopy_belief_propagation


if __name__ == '__main__':
    fg = FactorGraph()

    x1 = VNode("x1", Gaussian)
    x2 = VNode("x2", Gaussian)
    x3 = VNode("x3", Gaussian)

    f12 = FNode("f12", Gaussian([[0], [2]], [[1, 0.0], [0.0, 1]], x1, x2))
    f23 = FNode("f23", Gaussian([[1], [1]], [[1, 0.5], [0.5, 1]], x2, x3))
    f13 = FNode("f13", Gaussian([[2], [3]], [[1, 0.5], [0.5, 1]], x1, x3))

    fg.set_nodes([x1, x2, x3])
    fg.set_nodes([f12, f23, f13])

    fg.set_edge(f12, x1)
    fg.set_edge(f12, x2)
    fg.set_edge(f23, x2)
    fg.set_edge(f23, x3)
    fg.set_edge(f13, x1)
    fg.set_edge(f13, x3)

    b = loopy_belief_propagation(fg, 4, (x1,))

    for v, dists in b.items():
        print(v)
        for d in dists:
            print(d)

