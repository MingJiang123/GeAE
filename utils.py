import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm


def svm_classify(xs, ys, xt, yt, c=1, gamma='auto'):
    model = svm.SVC(C=c, kernel='rbf', gamma=gamma, decision_function_shape='ovr')
    ys = ys.ravel()
    yt = yt.ravel()
    model.fit(xs, ys)
    yt_pred = model.predict(xt)
    acc = accuracy_score(yt, yt_pred)
    return acc


def MBFeatures(array):
    n, p = array.shape
    array[np.abs(array) >= 0.3] = 1
    array[np.abs(array) < 0.3] = 0
    p = p - 1
    parents = np.copy(array[0:p, p:p + 1])
    children = np.copy(array[p:p + 1, 0:p])
    MB = parents + np.transpose(children)
    for j in range(p):
        if children[0, j] == 1:
            MB = MB + array[0:p, j:j + 1]
    MB[np.abs(MB) >= 1] = 1
    return parents, MB


def mysigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig


def getnew_xt(xt, w1, w2, b1, b2):
    layer_1 = mysigmoid(np.matmul(xt, w1) + b1)
    layer_2 = mysigmoid(np.matmul(layer_1, w2) + b2)
    return layer_2