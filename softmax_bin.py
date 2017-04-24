
import tensorflow as tf
import func_ml
import cv2
import os


def xaver_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


def train(obj_name, img_size):
    print "Loading training data ..."

    learning_rate = 0.001
    x_training = []
    y_training = []

    for category in range(2):
        dir_path = 'Pictures/' + obj_name + '/' + str(category)
        for f_name in os.listdir(dir_path):
            img_data = cv2.imread(dir_path + '/' + f_name, 0)
            resize_data = cv2.resize(img_data, (img_size, img_size), interpolation=cv2.INTER_AREA)
            x_training.append(resize_data.reshape(img_size * img_size))
            y_training.append(func_ml.expand0(category, 2))

    x = tf.placeholder("float", [None, img_size * img_size])
    y = tf.placeholder("float", [None, 2])  # 0-9 digits recognition => 10 classes

    W1 = tf.get_variable("W1", shape=[img_size * img_size, 2], initializer=xaver_init(img_size * img_size, 2))
    b1 = tf.Variable(tf.zeros([2]))
    activation = tf.add(tf.matmul(x, W1), b1)
    t1 = tf.nn.softmax(activation)

    # Minimize error using cross entropy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(activation, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Gradient Descent

    # Initializing the variables
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    print "Start training ..."
    # Training cycle
    for step in range(1000):
        sess.run(optimizer, feed_dict={x: x_training, y: y_training})
        if step % 10 == 0:
            ret = sess.run(t1, feed_dict={x: x_training})
            ret1 = func_ml.matrix_argmax(ret)
            ret2 = func_ml.matrix_argmax(y_training)
            acc1 = func_ml.acc(ret1, ret2)
            wgt_w = sess.run(W1)
            wgt_b = sess.run(b1)
            func_ml.save_csv('model/' + obj_name + '_w.csv', wgt_w)
            func_ml.save_csv('model/' + obj_name + '_b.csv', [wgt_b])
            print step, sess.run(cost, feed_dict={x: x_training, y: y_training}), acc1 * 100

    print ("Optimization Finished!")


if __name__ == '__main__':
    train('mouth', 30)
    # train('eyes', 30)
