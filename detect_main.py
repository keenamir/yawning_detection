
import cv2
from predict import face_classify

""" ----------------- Create the haar cascade ---------------- """
faceCascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('xml/haarcascade_eye_tree_eyeglasses.xml')
mouthCascade = cv2.CascadeClassifier('xml/haarcascade_mcs_mouth.xml')

""" -------------- Create the CNN classifier class ------------ """
mouth_class = face_classify('model/model_CNN_mouth')
eye_class = face_classify('model/model_CNN_eyes')

""" ------------------ Color and variable define -------------- """
color_red = (0, 0, 255)
color_blue = (255, 0, 0)
color_white = (255, 255, 255)
color_black = (0, 0, 0)
state = ['Close', 'Open']
cap = cv2.VideoCapture(0)

""" ---------------- camera read and process loop ------------- """
while True:
    # ------------ read image from camera and convert to gray -------------
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -------------------- get the camera video size ----------------------
    height, width = image.shape[:2]
    cv2.rectangle(image, (30, height - 30), (450, height - 30), color_black, 54)
    cv2.rectangle(image, (30, height - 30), (450, height - 30), color_white, 50)

    # ----- Detect face, mouth and eyes in the image using haar cascade ----
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100), flags=2)
    eyes = eyeCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=2)
    mouth = mouthCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=2)

    # -------------------- Calibrate the face data ------------------------
    if faces.__len__() > 0:
        face_data = [faces[0]]
    else:
        face_data = [[0, 0, 0, 0]]

    # -------------------- Calibrate the eyes data ------------------------
    eye_data = []
    for (x, y, w, h) in eyes:
        if x > face_data[0][0] and (x + w) < (face_data[0][0] + face_data[0][2]):           # out of x direction
            if y > face_data[0][1] and (y + h) < (face_data[0][1] + face_data[0][3]):       # out of y direction
                if (y + h/2) < (face_data[0][1] + face_data[0][3]/2):                       # below face
                    eye_data.append([x, y, w, h])

    # -------------------- Calibrate the mouth data ------------------------
    mouth_data = []
    for (x, y, w, h) in mouth:
        if x > face_data[0][0] and x + w < face_data[0][0] + face_data[0][2]:               # out of x direction
            if y + h/2 < face_data[0][1] + face_data[0][3]:                                 # out of y direction
                mouth_data.append([x, y, w, h])

    # ----------------- get mouth image data for deep learning --------------
    ret_mouth = 1
    for (x, y, w, h) in mouth_data:
        img_mouth = gray[y:y + h, x:x + w]
        ret_mouth = mouth_class.classify(img_mouth)

    ret_eye = [0, 0]
    eye_ind = 0
    for (x, y, w, h) in eye_data:
        img_eye = gray[y:y + h, x:x + w]
        ret_eye[eye_ind] = eye_class.classify(img_eye)
        eye_ind += 1

    # ------------------ print the result text on the image ------------------
    cv2.putText(image, "eye: " + state[ret_eye[0] and ret_eye[1]] + ", mouth: " + state[ret_mouth],
                (20, height - 20), cv2.FONT_HERSHEY_DUPLEX, 1, color_red, 2)

    # ------------------- Draw a rectangle around the faces -------------------
    for (x, y, w, h) in eye_data:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y, w, h) in mouth_data:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    for (x, y, w, h) in face_data:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # -------------------- display the result image ---------------------------
    cv2.imshow("Faces found", image)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

""" --------------------- device release and free --------------------------- """
cap.release()
cv2.destroyAllWindows()
