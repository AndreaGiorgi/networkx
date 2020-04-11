"""Shortest paths and path lengths using the A* ("A star") algorithm.
"""
from heapq import heappush, heappop
from itertools import count

import networkx as nx
from networkx.utils import not_implemented_for
from networkx.algorithms.shortest_paths.weighted import _weight_function

__all__ = ['beam_path', 'beam_path_length']


def beam_path(G, source, target, heuristic=None, weight='weight'):
    """Returns a list of nodes in a shortest path between source and target
    using the Greedy Best First algorithm.

    There may be more than one shortest path.  This returns only one.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path

    heuristic : function
       A function to evaluate the estimate of the distance
       from the a node to the target.  The function takes
       two nodes arguments and must return a number.

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight.
       Weight data are not used here, since greedy algorithm uses as function values only heuristic ones

    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.

    See Also
    --------
    shortest_path, dijkstra_path

    """
    if source not in G or target not in G:
        msg = f"Either source {source} or target {target} is not in G"
        raise nx.NodeNotFound(msg)

    if heuristic is None:
        # The default heuristic is h=0 - same as Dijkstra's algorithm
        def heuristic(u, v):
            return 0

    push = heappush
    pop = heappop

    # The queue stores priority, node, cost to reach, and parent.
    # Uses Python heapq to keep in priority order.
    # Add a counter to the queue to prevent the underlying heap from
    # attempting to compare the nodes themselves. The hash breaks ties in the
    # priority and is guaranteed unique for all nodes in the graph.

    c = count()
    queue = [(0, next(c), source, 0, None)]

    # Maps enqueued nodes to distance of discovered paths and the
    # computed heuristics to target. We avoid computing the heuristics
    # more than once and inserting the node into the queue too many times.
    enqueued = {}

    # Maps explored nodes to parent closest to the source.
    explored = {}

    # Maps path found to its cost
    beampath = {}

    # Stack for each node, used for explore different path
    stack = []

    # best_path returns the best one based on each total weight
    def best_path(path1, path2):
        return min(sum(weight(u, v, G[u][v]) for u, v in zip(path1[:-1], path1[1:])),
                   sum(weight(u, v, G[u][v]) for u, v in zip(path2[:-1], path2[1:])))

    def successors(v):
        return iter(sorted(G.neighbors(v), key=heuristic, reverse=True)[:2])

    while queue:
        # Pop the smallest item from queue.
        _, __, curnode, dist, parent = pop(queue)

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            if target in beampath:
                best = best_path(path, beampath[target])
                beampath[target] = best
                print(path)
            else:
                beampath[target] = path
                print(path)

        if curnode in explored:
            # Do not override the parent of starting node
            if explored[curnode] is None:
                continue

            # Skip bad paths that were enqueued before finding a better one
            queue_cost, h = enqueued[curnode]
            if queue_cost < dist:
                continue

        explored[curnode] = parent

        for neighbor in successors(curnode):
            print((list(successors(curnode))))
            node_cost = dist
            if neighbor in enqueued:
                queue_cost, h = enqueued[neighbor]
                # if queue_cost <= node_cost, a less costly path from the
                # neighbor to the source was already determined.
                # Therefore, we won't attempt to push this neighbor
                # to the queue
                if queue_cost <= node_cost:
                    continue
            else:
                h = heuristic(target)
            enqueued[neighbor] = node_cost, h
            push(queue, (node_cost + h, next(c), neighbor, node_cost, curnode))

    raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")


def beam_path_length(G, source, target, heuristic=None, weight='weight'):
    """Returns the length of the shortest path between source and target using
    the Beam Stack Search  algorithm.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path

    heuristic : function
       A function to evaluate the estimate of the distance
       from the a node to the target.  The function takes
       two nodes arguments and must return a number.

    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.

    See Also
    --------
    astar_path

    """
    if source not in G or target not in G:
        msg = f"Either source {source} or target {target} is not in G"
        raise nx.NodeNotFound(msg)

    weight = _weight_function(G, weight)
    path = beam_path(G, source, target, heuristic, weight)
    return sum(weight(u, v, G[u][v]) for u, v in zip(path[:-1], path[1:]))
