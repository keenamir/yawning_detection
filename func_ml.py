
import csv
import numpy as np
import tensorflow as tf
import cv2


def load_csv(filename):
    file_csv = open(filename, 'rb')
    reader = csv.reader(file_csv)
    data_csv = []
    for row_data in reader:
        data_csv.append(row_data)

    file_csv.close()
    return data_csv


def save_csv(filename, data):
    file_out = open(filename, 'wb')
    writer = csv.writer(file_out)
    writer.writerows(data)
    file_out.close()


def model2(W, b, x):
    out1 = np.matmul(x, W)
    out2 = np.add(out1, b)
    return out2


def matrix_argmax(data):
    ret_ind = []
    for item in data:
        ret_ind.append(np.argmax(item))
    return ret_ind


def acc(d1, d2):
    cnt = 0
    for i in xrange(d1.__len__()):
        if d1[i] == d2[i]:
            cnt += 1

    return float(cnt)/d1.__len__()


def expand0(number, width):
    s = np.zeros(width)
    s[number] = 1
    return s


def conv2(matrix):
    ret = []
    for i in matrix:
        item = []
        for j in i:
            item.append(float(j))
        ret.append(item)
    return ret


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
        l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 224, 224, 32)
                            strides=[1, 2, 2, 1], padding='SAME'))
        l1 = tf.nn.max_pool(l1a, ksize=[1, 3, 3, 1],              # l1 shape=(?, 38, 38, 32)
                            strides=[1, 3, 3, 1], padding='SAME')
        l1 = tf.nn.dropout(l1, p_keep_conv)

        l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 38, 38, 64)
                            strides=[1, 1, 1, 1], padding='SAME'))
        l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 19, 19, 64)
                            strides=[1, 2, 2, 1], padding='SAME')
        l2 = tf.nn.dropout(l2, p_keep_conv)

        l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 19, 19, 128)
                            strides=[1, 1, 1, 1], padding='SAME'))
        l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 10, 10, 128)
                            strides=[1, 2, 2, 1], padding='SAME')
        l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 12800)
        l3 = tf.nn.dropout(l3, p_keep_conv)

        l4 = tf.nn.relu(tf.matmul(l3, w4))
        l4 = tf.nn.dropout(l4, p_keep_hidden)

        pyx = tf.matmul(l4, w_o)
        return pyx
