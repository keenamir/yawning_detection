
import func_ml
import cv2
import tensorflow as tf


class face_classify:

    def __init__(self, model_name, size=30):
        """
            Initiation Function
            Same as the train_CNN "Configuration of CNN model" block
        """
        self.img_size = size

        tf.reset_default_graph()
        self.X = tf.placeholder("float", [None, self.img_size, self.img_size, 1])

        w1 = func_ml.init_weights([3, 3, 1, 16])                # 3x3x1 conv, 16 outputs
        w2 = func_ml.init_weights([3, 3, 16, 32])               # 3x3x16 conv, 32 outputs
        w3 = func_ml.init_weights([3, 3, 32, 64])               # 3x3x32 conv, 64 outputs
        w4 = func_ml.init_weights([64 * 2 * 2, 20])             # 64*2*2 input, 2000 outputs
        w5 = func_ml.init_weights([20, 2])                      # 2000 inputs, 1000 outputs (labels)

        self.p_keep_con = tf.placeholder("float")
        self.p_keep_hidden = tf.placeholder("float")
        py_x = func_ml.model(self.X, w1, w2, w3, w4, w5, self.p_keep_con, self.p_keep_hidden)

        self.predict_op = tf.argmax(py_x, 1)

        self.sess = tf.Session()
        init = tf.initialize_all_variables()
        self.sess.run(init)

        """ -------------- load the CNN model weights ------------ """
        saver = tf.train.Saver()
        saver.restore(self.sess, model_name)

    def load_image(self, img_name):
        """
            Load the image and return c0lor value
        """
        img_data = cv2.imread(img_name, 0)
        return img_data

    def classify(self, img_data):
        """
            resize the image and predict value using CNN model
        """
        resize_data = cv2.resize(img_data, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img_train = resize_data.reshape(self.img_size, self.img_size, 1)

        ret = self.sess.run(self.predict_op, feed_dict={self.X: [img_train], self.p_keep_con: 1, self.p_keep_hidden: 1})

        return ret[0]


if __name__ == "__main__":
    mouth_class = face_classify('model/model_CNN_mouth')                # create mouth classify class
    eye_class = face_classify('model/model_CNN_eyes')                   # create eye classify class

    eye_img = eye_class.load_image('Pictures/eyes/0/eye1_51.bmp')       # classify the eye using image
    result = eye_class.classify(eye_img)
    print 'eye', result
