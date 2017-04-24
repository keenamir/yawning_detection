# README #

This is the project detection of yawning using CNN.

### Package requirement ###

* Python 2.7
* TensorFlow 0.10
* OpenCV 3.2
* Numpy

### Progression of project ###

This project configured as 4 python scripts

* train_CNN.py

    This is the script for training using CNN(Convolution Neural Network)

    If we have training data, we should do the training using this script.

    The training data should exist on Pictures folder and we could set the path manually.

    If the training finished, the model weights are saved in model folder.

* predict.py

    This is the script predict the images whether close or open.

    This script use CNN model weights which were saved in model folder by traini_CNN.py

* detect_main.py

    This is the main script of this project, we display the webcam video and extract the eyes, mouth, then classify it.

    Opened and closed eyes and mouth classification is same as in predict.py.

    The classified result is shown as text on video.

* func_ml.py

    Some functions which used in previous scripts are saved in this script.
