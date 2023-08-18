import logging
import numpy as np
import networkx as nx
from sklearn import preprocessing
import pandas as pd


w_range = (0.5, 2.0)
def simulate_random_dag(d, degree, graph_type, w_range=(0.5, 2.0)):
    """Simulate random DAG with some expected degree.

    Args:
        d: number of nodes
        degree: expected node degree, in + out
        graph_type: {erdos-renyi, barabasi-albert, full}
        w_range: weight range +/- (low, high)

    Returns:
        W: weighted DAG
    """
    if graph_type == 'erdos-renyi':
        prob = float(degree) / (d - 1)
        B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
    elif graph_type == 'barabasi-albert':
        m = int(round(degree / 2))
        B = np.zeros([d, d])
        bag = [0]
        for ii in range(1, d):
            dest = np.random.choice(bag, size=m)
            for jj in dest:
                B[ii, jj] = 1
            bag.append(ii)
            bag.extend(dest)
    elif graph_type == 'full':  # ignore degree, only for experimental use
        B = np.tril(np.ones([d, d]), k=-1)
    else:
        raise ValueError('unknown graph type')
    # random permutation
    P = np.random.permutation(np.eye(d, d))  # permutes first axis only
    B_perm = P.T.dot(B).dot(P)
    U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
    U[np.random.rand(d, d) < 0.5] *= -1
    W = (B_perm != 0).astype(float) * U
    W[np.abs(W) >= 0.3] = 1
    W[np.abs(W) < 0.3] = 0
    return W


def mean_vec(low=0, high=100, d=50):
    # np.random.seed(123)
    return 0.01 * np.random.uniform(low, high, d)


def simulate_sem(W, n, noise_loc, mean_vec, mean_vec_noise, dataset_type='nonlinear_1'):
    """Simulate samples from SEM with specified type of noise.

    Args:
        W: weigthed DAG
        n: number of samples
        sem_type: {linear-gauss,linear-exp,linear-gumbel}
        noise_scale: scale parameter of noise distribution in linear SEM

    Returns:
        X: [n,d] sample matrix
    """
    G = nx.DiGraph(W)
    d = W.shape[0]
    X = np.random.multivariate_normal(mean_vec, 0.1 * np.eye(d), n)
    ordered_vertices = list(nx.topological_sort(G))
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = list(G.predecessors(j))
        if dataset_type == 'nonlinear_1':
            eta = X[:, j] + np.cos(X[:, parents] + 1).dot(W[parents, j])
        elif dataset_type == 'linear_1':
            eta = X[:, j] + 0.2 * (X[:, parents] + 1).dot(W[parents, j])
        else:
            raise ValueError('Unknown linear data type')
        X[:, j] = eta + np.random.normal(loc=noise_loc, scale=1, size=n)
    X_noise = np.random.multivariate_normal(mean_vec_noise, 0.1 * np.eye(len(mean_vec_noise)), n)
    # X_minmax = preprocessing.MinMaxScaler().fit_transform(X)
    # X_stand = preprocessing.StandardScaler().fit_transform(X)
    X_all = np.hstack((X_noise, X))
    return X_all


if __name__ == '__main__':
    dir = ['50non', '100non', '150non', '200non', '250non', '300non']
    for i in range(6):
        n, d, d_noise = 500, 51 + 50 * i, 30
        graph_type, degree, sem_type = 'erdos-renyi', 10, 'linear-gauss'
        W = simulate_random_dag(d, degree, graph_type)
        mean_vec1 = mean_vec(low=0, high=50, d=d)
        mean_vec2 = mean_vec(low=100, high=150, d=d)
        mean_vec1_noise = mean_vec(low=0, high=10, d=d_noise)
        mean_vec2_noise = mean_vec(low=50, high=60, d=d_noise)
        Xs_1 = simulate_sem(W, n, 0, mean_vec1, mean_vec1_noise, dataset_type='nonlinear_1')
        Xs_2 = simulate_sem(W, n, 0, mean_vec2, mean_vec1_noise, dataset_type='nonlinear_1')
        Xt_1 = simulate_sem(W, n, 0.5, mean_vec1, mean_vec2_noise, dataset_type='nonlinear_1')
        Xt_2 = simulate_sem(W, n, 0.5, mean_vec2, mean_vec2_noise, dataset_type='nonlinear_1')
        Xs = np.vstack((Xs_1[:, :-1], Xs_2[:, :-1]))
        Xt = np.vstack((Xt_1[:, :-1], Xt_2[:, :-1]))
        y = np.hstack((np.array([0 for _ in range(n)]), np.array([1 for _ in range(n)])))
        pd.DataFrame(Xs).to_csv('../data/simulated-dataset/datasets2/xs_' + dir[i] + '.csv', index=False)
        pd.DataFrame(Xt).to_csv('../data/simulated-dataset/datasets2/xt_' + dir[i] + '.csv', index=False)
        pd.DataFrame(y).to_csv('../data/simulated-dataset/datasets2/ys_' + dir[i] + '.csv', index=False)
        pd.DataFrame(y).to_csv('../data/simulated-dataset/datasets2/yt_' + dir[i] + '.csv', index=False)

