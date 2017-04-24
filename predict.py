
import func_ml
import cv2
import numpy as np


class face_classify:

    def __init__(self, size=30):
        """
            Initiation Function
        """
        self.img_size = size
        self.eye_w = func_ml.load_csv("model/eyes_w.csv")
        self.eye_b = func_ml.load_csv("model/eyes_b.csv")
        self.mouth_w = func_ml.load_csv("model/mouth_w.csv")
        self.mouth_b = func_ml.load_csv("model/mouth_b.csv")

    def load_image(self, img_name):
        """
            Load the image and return c0lor value
        """
        img_data = cv2.imread(img_name)
        return img_data

    def classify(self, img_data, obj_name):

        gray_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
        resize_data = cv2.resize(gray_data, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img_train = resize_data.reshape(self.img_size * self.img_size)

        if obj_name == 'eye':
            ret = func_ml.model2(func_ml.conv2(self.eye_w), func_ml.conv2(self.eye_b), img_train)
        elif obj_name == 'mouth':
            ret = func_ml.model2(func_ml.conv2(self.mouth_w), func_ml.conv2(self.mouth_b), img_train)

        return np.argmax(ret)


if __name__ == "__main__":
    my_class = face_classify()
    color_img = my_class.load_image('Pictures/mouth/1/mouth_2.bmp')
    object_name = 'mouth'
    result = my_class.classify(color_img, object_name)
    print object_name, result
