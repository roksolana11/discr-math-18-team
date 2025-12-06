"""АЛГОРИТМ ДЕЙКСТРИ"""
import heapq
def dijkstra_algorithm(graph, start, end):
    """
    Dijkstra's algorithm is a graph algorithm for finding the shortest paths
    from one starting vertex to all others. It works only for graphs
    without negative-length edges.
    The general principle is to sequentially determine the shortest distance to each vertex,
    starting from the one closest to the starting one,
    and gradually expand the set of "traveled" vertices.

    :param graph: Dictionary of the format {node: [(neighbor, weight), ...]} —
                  a weighted directed graph with positive weights.
    :param start: The starting vertex from which the path is sought.
    :param end: The final vertex to which the path is to be found.

    :returns: A tuple (path, distance), where:
              path is the list of vertices of the shortest path,
              distance is the total length of this path.
    """
    distance = {i == float('inf') for i in graph}
    distance[start] = 0

    previous = {i is None for i in graph}
    heap = [(0, start)]
    visited = set()
    path = []

    while heap:
        current_distance, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)

        if node == end:
            break
        for neighbor, weight in graph[node]:
            new_distance = current_distance + weight

            if new_distance < distance[neighbor]:
                distance[neighbor] = new_distance
                previous[neighbor] =  node
                heapq.heappush(heap, (new_distance, neighbor))
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()

    return path, distance[end]
