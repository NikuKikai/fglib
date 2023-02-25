from typing import List, Dict, Tuple
import numpy as np
import cv2
from fglib import graphs, nodes, inference, rv
from fglib.nodes import VNode, FNode, NodeType
from fglib.edges import Edge
from fglib.rv import Gaussian
import networkx as nx
import pygame as pg
import pygame.locals as pgl
import sys


class ObstacleMap_Deprecated:  # NOTE cv2.distanceTransform is not precise!
    def __init__(self, w, h, x0=0, y0=0) -> None:
        self.x0 = x0
        self.y0 = y0
        self.w = w
        self.h = h
        self.arr = np.zeros((h, w), np.uint8)
        self.update_distmap()

    def update_distmap(self):
        d = cv2.distanceTransform(self.arr, cv2.DIST_L2, 5)
        d_ = cv2.distanceTransform(np.where(self.arr>0, 0, 1).astype(np.uint8), cv2.DIST_L2, 3)
        self.distmap = d_ - d

    def get_d_grad(self, x, y) -> Tuple[float, float, float]:
        from scipy.interpolate import RegularGridInterpolator
        y, x = y-self.y0, x-self.x0
        diffy = np.diff(self.distmap, axis=0)
        diffx = np.diff(self.distmap, axis=1)

        interp = RegularGridInterpolator((np.arange(self.h), np.arange(self.w)), self.distmap)
        interp_grady = RegularGridInterpolator((np.arange(self.h-1), np.arange(self.w)), diffy)
        interp_gradx = RegularGridInterpolator((np.arange(self.h), np.arange(self.w-1)), diffx)
        return interp((y, x)), interp_gradx((y, x-0.5)), interp_grady((y-0.5, x))


class ObstacleMap:
    def __init__(self) -> None:
        self.objects = {}

    def set_circle(self, name: str, centerx, centery, radius):
        o = {'type': 'circle', 'name': name, 'centerx': centerx, 'centery': centery, 'radius': radius}
        self.objects[name] = o

    def get_d_grad(self, x, y) -> Tuple[float, float, float]:
        mindist = np.inf
        mino = None
        for o in self.objects.values():
            if o['type'] == 'circle':
                ox, oy, r = o['centerx'], o['centery'], o['radius']
                d = np.sqrt((x - ox)**2 + (y - oy)**2) - r
                if d < mindist:
                    mindist = d
                    mino = o
        if mino is None:
            return np.inf, 0, 0
        if mino['type'] == 'circle':
            ox, oy = o['centerx'], o['centery']
            dx, dy = x - ox, y - oy
            mag = np.sqrt(dx**2 + dy**2)
            return mindist, dx/mag, dy/mag


class DynaFNode(FNode):
    def update_factor(self, x0: Gaussian, x1: Gaussian, dt: float = 1, precision: float = 1):
        # NOTE target: ||h(x) - z)||2 -> 0
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


class ObstacleFNode(FNode):
    def __init__(self, label, factor=None, env: ObstacleMap = None, safe_radius: float = 4, precision: float = 10):
        super().__init__(label, factor)
        self._env = env
        self._safe_radius = safe_radius
        self._precision = precision

    def update_factor(self, x: Gaussian):
        # target: ||h(x) - z)||2 -> 0
        z = np.zeros((1, 1))
        v = x.mean  # [4, 1]

        distance, distance_gradx, distance_grady = self._env.get_d_grad(v[0, 0], v[1, 0])
        distance -= self._safe_radius

        h = np.array([[max(0, 1 - distance / self._safe_radius)]])
        jacob = np.array([[-distance_gradx/self._safe_radius, -distance_grady/self._safe_radius, 0, 0]])  # [1, 4]
        precision = np.identity(1) * self._precision

        w = jacob.T @ precision @ jacob
        wm = jacob.T @ precision @ (jacob @ v + z - h)
        self.factor = Gaussian.inf_form(w, wm, *x.dim)


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
            elif type(f) is ObstacleFNode:
                f.update_factor(vs[0].belief)

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
    omap = ObstacleMap()
    omap.set_circle('', 145, 85, 30)
    robot_r = 5

    fg = graphs.FactorGraph()

    x0 = VNode("x0", Gaussian(np.random.rand(4, 1), np.diag([10] * 4), 'x0.x', 'x0.y', 'x0.vx', 'x0.vy'))
    x1 = VNode("x1", Gaussian(np.random.rand(4, 1), np.diag([10] * 4), 'x1.x', 'x1.y', 'x1.vx', 'x1.vy'))
    x2 = VNode("x2", Gaussian(np.random.rand(4, 1), np.diag([10] * 4), 'x2.x', 'x2.y', 'x2.vx', 'x2.vy'))
    x3 = VNode("x3", Gaussian(np.random.rand(4, 1), np.diag([10] * 4), 'x3.x', 'x3.y', 'x3.vx', 'x3.vy'))

    f0 = FNode("f0", Gaussian(np.array([0, 0, 100, 50])[..., None], np.diag([.01] * 4), 'x0.x', 'x0.y', 'x0.vx', 'x0.vy'))
    f01 = DynaFNode("f01", None)
    f12 = DynaFNode("f12", None)
    f23 = DynaFNode("f23", None)
    f3 = FNode("f3", Gaussian(np.array([350, 100, 0, 0])[..., None], np.diag([.1] * 4), 'x3.x', 'x3.y', 'x3.vx', 'x3.vy'))
    fo1 = ObstacleFNode('fo1', env=omap, safe_radius=robot_r, precision=100)
    fo2 = ObstacleFNode('fo2', env=omap, safe_radius=robot_r, precision=100)

    fg.set_nodes([x0, x1, x2, x3])
    fg.set_nodes([f0, f01, f12, f23, f3, fo1, fo2])

    fg.set_edge(f0, x0)
    fg.set_edge(f3, x3)
    fg.set_edge(f01, x0)
    fg.set_edge(f01, x1)
    fg.set_edge(f12, x1)
    fg.set_edge(f12, x2)
    fg.set_edge(f23, x2)
    fg.set_edge(f23, x3)
    fg.set_edge(fo1, x1)
    fg.set_edge(fo2, x2)


    pg.init()
    surf = pg.display.set_mode((1000, 800))
    while True:
        surf.fill((0, 0, 0))

        # Draw obstacles
        for o in omap.objects.values():
            if o['type'] == 'circle':
                pg.draw.circle(surf, (222, 0, 0), (o['centerx']+10, o['centery']+10), o['radius'], 1)

        bs = loopy_belief_propagation(fg, 1)
        for vnode, p in bs.items():
            color = {'x0': (255, 255, 255), 'x1': (222, 222, 222), 'x2': (188, 188, 188), 'x3': (155, 155, 155)}[str(vnode)]
            x, y, vx, vy = p.mean[:, 0]
            pg.draw.circle(surf, color, (x+10, y+10), robot_r, 1)

        for event in pg.event.get():
            if event.type == pgl.QUIT:
                pg.quit()
                sys.exit()
        pg.time.wait(500)
        pg.display.update()


