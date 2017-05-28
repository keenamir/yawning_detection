
import tensorflow as tf
import func_ml
import cv2
import os
import numpy as np


def train(obj_name, img_size, para_rate):
    """
        Convolution Neural Network training block
    """

    """ ---------- load the training data from training images ------------------ """
    print "Loading training data ..."

    train_x = []
    train_y = []

    for category in range(2):
        dir_path = 'Pictures/' + obj_name + '/' + str(category)
        for f_name in os.listdir(dir_path):
            img_data = cv2.imread(dir_path + '/' + f_name, 0)                                       # image read
            resize_data = cv2.resize(img_data, (img_size, img_size), interpolation=cv2.INTER_AREA)  # image resize
            train_x.append(resize_data.reshape(img_size, img_size, 1))                              # reshape and add
            train_y.append(func_ml.expand0(category, 2))

    """ ---------------------- Configuration of CNN model ------------------------- """
    print ("Configuration of CNN model ...")
    X = tf.placeholder("float", [None, img_size, img_size, 1])                  # define x data of training
    Y = tf.placeholder("float", [None, 2])                                      # define y data of training

    w1 = func_ml.init_weights([3, 3, 1, 16])                                    # 3x3x1 conv, 16 outputs
    w2 = func_ml.init_weights([3, 3, 16, 32])                                   # 3x3x16 conv, 32 outputs
    w3 = func_ml.init_weights([3, 3, 32, 64])                                   # 3x3x32 conv, 64 outputs
    w4 = func_ml.init_weights([64 * 2 * 2, 20])                                 # 64*2*2 input, 2000 outputs
    w5 = func_ml.init_weights([20, 2])                                          # 2000 inputs, 1000 outputs (labels)

    p_keep_con = tf.placeholder("float")                                        # convolution parameter
    p_keep_hidden = tf.placeholder("float")                                     # hidden parameter
    py_x = func_ml.model(X, w1, w2, w3, w4, w5, p_keep_con, p_keep_hidden)      # define model output

    cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))  # define cost function
    train_op = tf.train.AdamOptimizer(para_rate).minimize(cost_op)              # using AdamOptimizer for optimize
    predict_op = tf.argmax(py_x, 1)                                             # define predict value

    sess = tf.Session()                                                         # session define
    init = tf.initialize_all_variables()                                        # session initialize
    sess.run(init)
    saver = tf.train.Saver()                                                    # session saver define

    """ ------------------------------ training data ---------------------------- """
    print "Training data ..."

    for step_i in range(3000):
        # train the CNN model and return cost value
        ret = sess.run([train_op, cost_op], feed_dict={X: train_x, Y: train_y, p_keep_con: 0.8, p_keep_hidden: 0.8})


    print ("Optimization Finished!")

if __name__ == '__main__':
    train('mouth', 30, 0.0001)
    train('eyes', 30, 0.0001)
