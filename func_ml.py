
import xlwt
import csv
import numpy as np
# import tensorflow as tf
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
