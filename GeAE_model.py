import tensorflow as tf
import numpy as np
from utils import MBFeatures


def model(xs, ys, c_num=2, iter1_max=1000, iter2_max=10, lambda1=1.0, lambda2=1.0, lambda3=1.0):
    tf.reset_default_graph()
    n, d = xs.shape
    n_hidden_1 = 50
    n_hidden_2 = 50
    n_hidden_3 = 51
    n_hidden_4 = 51
    gamma = 0.25
    beta = 10
    learning_rate = 0.001
    display_step = 100
    tol = 1e-8

    X = tf.placeholder('float', shape=[None, d])
    y = tf.placeholder('float', shape=[None, 1])
    alpha = tf.placeholder(tf.dtypes.float32)
    rho = tf.placeholder(tf.dtypes.float32)
    W_init = tf.Variable(tf.random.uniform([n_hidden_4, n_hidden_4], minval=-0.1, maxval=0.1), dtype=tf.dtypes.float32)
    W_mb = tf.placeholder('float', shape=[1, n_hidden_1])


    def _preprocess_graph(W):
        # Mask the diagonal entries of graph
        return tf.matrix_set_diag(W, tf.zeros(W.shape[0], dtype=tf.dtypes.float32))
    W = _preprocess_graph(W_init)


    weights = {'encoder_w1': tf.Variable(tf.random_normal(shape=[d, n_hidden_1])),
               'encoder_w2': tf.Variable(tf.random_normal(shape=[n_hidden_1, n_hidden_2])),
               'encoder_w3': tf.Variable(tf.random_normal(shape=[n_hidden_2 + 1, n_hidden_3])),
               'encoder_w4': tf.Variable(tf.random_normal(shape=[n_hidden_3, n_hidden_4])),
               'decoder_w1': tf.Variable(tf.random_normal(shape=[n_hidden_4, n_hidden_3])),
               'decoder_w2': tf.Variable(tf.random_normal(shape=[n_hidden_3, n_hidden_2 + 1])),
               'decoder_w3': tf.Variable(tf.random_normal(shape=[n_hidden_2, n_hidden_1])),
               'decoder_w4': tf.Variable(tf.random_normal(shape=[n_hidden_1, d])),
               'prediction_w': tf.Variable(tf.random_normal(shape=[n_hidden_2, c_num]))
               }

    biases = {'encoder_b1': tf.Variable(tf.random_normal(shape=[n_hidden_1])),
              'encoder_b2': tf.Variable(tf.random_normal(shape=[n_hidden_2])),
              'encoder_b3': tf.Variable(tf.random_normal(shape=[n_hidden_3])),
              'encoder_b4': tf.Variable(tf.random_normal(shape=[n_hidden_4])),
              'decoder_b1': tf.Variable(tf.random_normal(shape=[n_hidden_3])),
              'decoder_b2': tf.Variable(tf.random_normal(shape=[n_hidden_2 + 1])),
              'decoder_b3': tf.Variable(tf.random_normal(shape=[n_hidden_1])),
              'decoder_b4': tf.Variable(tf.random_normal(shape=[d])),
              'prediction_b': tf.Variable(tf.random_normal(shape=[c_num]))
              }


    def encoder(x, y):
        layer_1 = tf.sigmoid(tf.add(tf.matmul(x, weights['encoder_w1']), biases['encoder_b1']))
        layer_2 = tf.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_w2']), biases['encoder_b2']))
        layer_2_1 = tf.concat([layer_2, y], axis=1)
        layer_3 = tf.sigmoid(tf.add(tf.matmul(layer_2_1, weights['encoder_w3']), biases['encoder_b3']))
        layer_4 = tf.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_w4']), biases['encoder_b4']))
        return layer_4, layer_2, layer_2_1


    def decoder(x):
        layer_1 = tf.sigmoid(tf.add(tf.matmul(x, weights['decoder_w1']), biases['decoder_b1']))
        layer_2 = tf.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_w2']), biases['decoder_b2']))
        layer_2_1 = layer_2[:, :-1]
        layer_3 = tf.sigmoid(tf.add(tf.matmul(layer_2_1, weights['decoder_w3']), biases['decoder_b3']))
        layer_4 = tf.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_w4']), biases['decoder_b4']))
        return layer_4, layer_3, layer_2


    X_encoder, X_encoder_hl2, X_encoder_mid = encoder(X, y)
    X_encoder_y = tf.matmul(X_encoder, W)
    X_decoder, X_decoder_hl2, X_decoder_mid = decoder(X_encoder_y)

    loss_recon = tf.reduce_mean(tf.pow(X_decoder-X, 2)) / 2
    loss_causal = tf.reduce_mean(tf.pow(X_decoder_mid - X_encoder_mid, 2)) / 2


    loss_l2w = 0
    for key, value in weights.items():
        loss_l2w += tf.reduce_mean(tf.square(value))


    ys_pred = tf.add(tf.matmul(tf.multiply(X_encoder_hl2, W_mb), weights['prediction_w']), biases['prediction_b'])
    loss_cross = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y[:, -1], dtype=tf.int32), logits=ys_pred))
    h = tf.linalg.trace(tf.linalg.expm(W * W)) - (d+1)
    loss_fun = loss_recon + lambda1 * loss_causal + lambda2 * loss_l2w + lambda3 * loss_cross +\
               alpha * h + 0.5 * rho * h * h
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_fun)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        alpha_val = 0.0
        rho_val = 1.0
        h_val = 0
        W_mb_val = np.ones([1, n_hidden_1])

        for i in range(iter2_max):
            print('the number of iter2 is:', i)
            loss_1 = 0
            for j in range(iter1_max):
                _, loss, l_r, l_cr, l2_w, l_c, W_curr, h_curr = sess.run([optimizer, loss_fun, loss_recon, loss_causal,
                                                                     loss_l2w, loss_cross, W, h],
                                                                    feed_dict={X: xs, y: ys, alpha: alpha_val, rho: rho_val, W_mb: W_mb_val})
                if np.abs(loss-loss_1) <= tol:
                    break
                loss_1 = loss
                if j % display_step == 0 or j == 1:
                    print('the number of iter1 is:', j)
                    print('loss is:', loss_1)
            X_new, w1, w2, b1, b2, W_new, h_new = sess.run([X_encoder_hl2, weights['encoder_w1'], weights['encoder_w2'], biases['encoder_b1'], biases['encoder_b2'], W, h], feed_dict={X: xs, y: ys, alpha: alpha_val, rho: rho_val, W_mb: W_mb_val})
            W_new1 = np.copy(W_new)
            _, W_mb_array = MBFeatures(W_new1)
            W_mb_val = W_mb_array.T
            alpha_new = alpha_val + rho_val * h_new
            if np.abs(h_new) >= gamma * np.abs(h_val):
                rho_new = beta * rho_val
            else:
                rho_new = rho_val
            alpha_val = alpha_new
            rho_val = rho_new
            h_val = h_new
        X_new, w1, w2, b1, b2, W_new, h_new = sess.run([X_encoder_hl2, weights['encoder_w1'], weights['encoder_w2'], biases['encoder_b1'], biases['encoder_b2'], W, h], feed_dict={X: xs, y: ys, alpha: alpha_val, rho: rho_val, W_mb: W_mb_val})
    pa, mb = MBFeatures(W_new)
    idx = np.argwhere(mb == 1)[:, 0]
    return X_new[:, idx], idx, w1, w2, b1, b2