import numpy as np



def FloydWarshallAllPaths(A, tol=1e-6):
    num_nodes = A.shape[0]

    dist = np.ones((num_nodes, num_nodes)) * np.inf
    prevs = np.empty((num_nodes, num_nodes), dtype=object)
    for i in range(num_nodes):
        for j in range(num_nodes):
            prevs[i, j] = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            dist[i, j] = A[i, j]
            if A[i, j] != np.inf:
                prevs[i, j] = [i]

    # Floyd-Warshall
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
                    prevs[i, j] = prevs[k, j].copy()
                elif dist[i, j] == dist[i, k] + dist[k, j] and np.isfinite(dist[i, j]):
                    new_prevs = []
                    for prev in prevs[k, j]:
                        if prev not in prevs[i, j]:
                            new_prevs.append(prev)

                    prevs[i, j].extend(new_prevs)

    return dist, prevs

def all_paths(prevs, i, j):
    if not prevs[i, j]:
        return []

    return enumerate_paths(prevs, i, j, [j])

# def enumerate_paths(prevs, i, j, curr_path):
#     if len(prevs[i, j]) == 1 and prevs[i, j][0] == i:
#         stem = [i]
#         stem.extend(curr_path)
#         return [stem]
#
#     all_paths = []
#     for pdx, prev in enumerate(prevs[i, j]):
#         stem = [prev]
#         stem.extend(curr_path)
#         all_children = enumerate_paths(prevs, i, prev, stem)
#         for child in all_children:
#             all_paths.append(child)
#
#     return all_paths

def enumerate_paths(prevs, i, j, curr_path):
    all_paths = []
    for pdx, prev in enumerate(prevs[i, j]):
        stem = [prev]
        stem.extend(curr_path)
        if prev == i:
            all_children = [stem]
        else:
            all_children = enumerate_paths(prevs, i, prev, stem)

        for child in all_children:
            all_paths.append(child)

    return all_paths

def modify_unweighted_A(unweighted_A):
    weighted_A = np.copy(unweighted_A)
    weighted_A[unweighted_A == 0] = np.inf

    return weighted_A

def betweenness_centrality(A, convert_unweighted=False):
    if convert_unweighted:
        A = modify_unweighted_A(A)

    num_nodes = A.shape[0]

    dists, prevs = FloydWarshallAllPaths(A)

    bc = np.zeros(num_nodes)
    for vert in range(num_nodes):
        tot_paths = 0
        tot_vert_paths = 0
        for i in range(num_nodes):
            if i == vert:
                continue
            for j in range(num_nodes):
                if j == vert:
                    continue

                all_shortest_paths_ij = all_paths(prevs, i, j)
                tot_paths += len(all_shortest_paths_ij)

                for shortest_path_ij in all_shortest_paths_ij:
                    if vert in shortest_path_ij:
                        tot_vert_paths += 1

        bc[vert] = tot_vert_paths / tot_paths

    return bc

def resolvent_centrality(A, gamma):
    assert 1 / np.sort(np.linalg.eig(A)[0])[-1] > gamma

    num_nodes = A.shape[0]
    return np.diag(np.linalg.inv(np.eye(num_nodes) - gamma * A))


if __name__ == '__main__':

    print('Diamond graph')
    adjmat = np.array([[np.inf, 1, 1, np.inf], [1, np.inf, np.inf, 1], [1, np.inf, np.inf, 1], [np.inf, 1, 1, np.inf]])
    dists, prevs = FloydWarshallAllPaths(adjmat)
    print('\t paths from 0->3', all_paths(prevs, 0, 3))
    print('\t paths from 1->2', all_paths(prevs, 1, 2))
    print('\t paths from 0->0', all_paths(prevs, 0, 0))
    print('\t BC', betweenness_centrality((adjmat)))
    unweighted_adjmat = np.zeros_like(adjmat)
    unweighted_adjmat[adjmat == 1] = 1
    print('\t RC', resolvent_centrality(unweighted_adjmat, gamma=0.1))

    print('Snake graph')
    adjmat = np.inf * np.ones((7, 7))
    adjmat[0, 1] = 1
    adjmat[0, 2] = 1

    adjmat[1, 0] = 1
    adjmat[1, 3] = 1

    adjmat[2, 0] = 1
    adjmat[2, 3] = 1

    adjmat[3, 1] = 1
    adjmat[3, 2] = 1
    adjmat[3, 4] = 1
    adjmat[3, 5] = 1

    adjmat[4, 3] = 1
    adjmat[4, 6] = 1

    adjmat[5, 3] = 1
    adjmat[5, 6] = 1

    adjmat[6, 4] = 1
    adjmat[6, 5] = 1

    dists, prevs = FloydWarshallAllPaths(adjmat)
    print('\t paths 0->6', all_paths(prevs, 0, 6))

    print('Branch diamond graph')
    adjmat = np.inf * np.ones((10, 10))
    adjmat[0, 1] = 1
    adjmat[0, 5] = 1

    adjmat[1, 0] = 1
    adjmat[1, 2] = 1
    adjmat[1, 3] = 1

    adjmat[2, 1] = 1
    adjmat[2, 4] = 1

    adjmat[3, 1] = 1
    adjmat[3, 4] = 1

    adjmat[4, 2] = 1
    adjmat[4, 3] = 1
    adjmat[4, 9] = 1

    adjmat[5, 0] = 1
    adjmat[5, 6] = 1
    adjmat[5, 7] = 1

    adjmat[6, 5] = 1
    adjmat[6, 8] = 1

    adjmat[7, 5] = 1
    adjmat[7, 8] = 1

    adjmat[8, 6] = 1
    adjmat[8, 7] = 1
    adjmat[8, 9] = 1

    adjmat[9, 4] = 1
    adjmat[9, 8] = 1

    dists, prevs = FloydWarshallAllPaths(adjmat)
    print('\t paths 0->9', all_paths(prevs, 0, 9))
    print('\t BC', betweenness_centrality(adjmat))

    print('Redundancy graph')
    adjmat = np.array([[np.inf, 2, 1], [2, np.inf, 1], [1, 1, np.inf]])
    print(betweenness_centrality(adjmat, convert_unweighted=False))

    print('Bottleneck chamber graph, multiselfloop')
    d = np.load('bottleneck_A_multi.npz')
    adjmat = d['A']
    print(betweenness_centrality(adjmat, convert_unweighted=True))




