import numpy as np
import networkx as nx
from GeAE_model import model
from utils import svm_classify, getnew_xt


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
    return W


def simulate_sem(W, n, sem_type, noise_loc=0.0, noise_scale=1.0, dataset_type='nonlinear_1'):
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
    X = np.zeros([n, d])
    ordered_vertices = list(nx.topological_sort(G))
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = list(G.predecessors(j))
        # print(parents)
        if dataset_type == 'nonlinear_1':
            eta = np.cos(X[:, parents] + 1).dot(W[parents, j])
        elif dataset_type == 'nonlinear_2':
            eta = np.sin(X[:, parents] + 1).dot(W[parents, j])
        elif dataset_type == 'linear_1':
            eta = (X[:, parents] + 1).dot(W[parents, j])
        elif dataset_type == 'linear_2':
            eta = (X[:, parents] + 2).dot(W[parents, j])
        else:
            raise ValueError('Unknown linear data type')

        if sem_type == 'linear-gauss':
            X[:, j] = eta + np.random.normal(loc=noise_loc, scale=noise_scale, size=n)
        else:
            raise NotImplementedError
    return X


if __name__ == '__main__':
    n, d = 500, 51
    graph_type, degree, sem_type = 'erdos-renyi', 10, 'linear-gauss'
    noise_loc = [0.0, 0.5]
    noise_scale = 1.0
    W = simulate_random_dag(d, degree, graph_type)
    Xs_1 = simulate_sem(W, n, sem_type, noise_loc=noise_loc[0], noise_scale=1.0, dataset_type='nonlinear_1')
    Xs_2 = simulate_sem(W, n, sem_type, noise_loc=noise_loc[1], noise_scale=1.0, dataset_type='nonlinear_1')
    Xt_1 = simulate_sem(W, n, sem_type, noise_loc=noise_loc[0], noise_scale=1.0, dataset_type='nonlinear_2')
    Xt_2 = simulate_sem(W, n, sem_type, noise_loc=noise_loc[1], noise_scale=1.0, dataset_type='nonlinear_2')
    Xs = np.vstack((Xs_1[:, :-1], Xs_2[:, :-1]))
    Xt = np.vstack((Xt_1[:, :-1], Xt_2[:, :-1]))
    y = np.hstack((np.array([0 for _ in range(n)]), np.array([1 for _ in range(n)]))).reshape(-1, 1)
    xs_new, idx, w1, w2, b1, b2 = model(xs=Xs, ys=y, iter1_max=3000, iter2_max=10, lambda1=1, lambda2=1, lambda3=1)
    xt_new = getnew_xt(Xt, w1, w2, b1, b2)[:, idx]
    acc = svm_classify(xs=xs_new, ys=y, xt=xt_new, yt=y)
    print(acc)

