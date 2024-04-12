import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
from itertools import combinations

seed = 12345
rng = np.random.default_rng(seed)


class Node:
    def __init__(self, id):
        self.id = id
        self.x, self.y = rng.random(2)
        self.xforce = 0.0
        self.yforce = 0.0


class Link:
    def __init__(self, source, target, value):
        self.source = source
        self.target = target
        self.value = value


def parse_xml(xmlfile):

    # create element tree object
    tree = ET.parse(xmlfile)

    root = tree.getroot()
    meta = root.find('MetaNetwork')

    nodes = meta.find('nodes')

    graph = {'nodes': {}, 'links': []}
    for node in nodes.find('nodeclass'):
        id = node.get('id')
        graph['nodes'][id] = Node(id)

    for network in meta.find('networks'):
        if network.get('id') == 'ZACHC':
            for link in network.findall('link'):
                source = link.get('source')
                target = link.get('target')
                value = link.get('value')
                graph['links'].append(Link(source, target, value))

    return graph


# https://en.wikipedia.org/wiki/Force-directed_graph_drawing
def apply_force(graph):
    coulomb = -2.0
    spring = 10.0
    dt = 1e-3

    maxiter = 10000
    tol = 1e-4
    for iter in range(maxiter):

        for node in graph['nodes'].values():
            node.xforce = 0.0
            node.yforce = 0.0

        # repulsive force = k * qa * qb / r^2 = coulomb / r^2
        for akey, bkey in combinations(graph['nodes'], 2):
            a = graph['nodes'][akey]
            b = graph['nodes'][bkey]

            dx = b.x - a.x
            dy = b.y - a.y
            dist2 = dx * dx + dy * dy
            dist = np.sqrt(dist2)

            magnitude = coulomb / dist2
            xmag = magnitude * dx / dist
            ymag = magnitude * dy / dist

            a.xforce += xmag
            a.yforce += ymag

            b.xforce -= xmag
            b.yforce -= ymag

        # attractive force = k * x
        for link in graph['links']:
            source = graph['nodes'][link.source]
            target = graph['nodes'][link.target]

            dx = target.x - source.x
            dy = target.y - source.y
            dist2 = dx * dx + dy * dy
            dist = np.sqrt(dist2)

            magnitude = spring * dist
            xmag = magnitude * dx / dist
            ymag = magnitude * dy / dist

            source.xforce += xmag
            source.yforce += ymag

        # apply force
        residual = 0.0
        for node in graph['nodes'].values():
            xold = node.x
            yold = node.y

            node.x += node.xforce * dt
            node.y += node.yforce * dt

            dx = node.x - xold
            dy = node.y - yold
            delta2 = dx * dx + dy * dy
            residual += delta2

        if residual < tol:
            print('converged after {} iterations'.format(iter))
            break


def main():
    file = 'zachary.xml'
    graph = parse_xml(file)
    apply_force(graph)

    fig, ax = plt.subplots()

    for link in graph['links']:
        source = graph['nodes'][link.source]
        target = graph['nodes'][link.target]
        ax.plot([source.x, target.x], [source.y, target.y], 'r')

    for _, node in graph['nodes'].items():
        ax.plot(node.x, node.y, 'o',
                markersize=20, markerfacecolor='cornflowerblue', markeredgecolor='tab:blue')
        ax.text(node.x, node.y, node.id, horizontalalignment='center', verticalalignment='center')

    plt.show()


if __name__ == '__main__':
    main()
