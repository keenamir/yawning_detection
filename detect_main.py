
import cv2
from predict import face_classify

# Create the haar cascade
faceCascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('xml/haarcascade_eye_tree_eyeglasses.xml')
mouthCascade = cv2.CascadeClassifier('xml/haarcascade_mcs_mouth.xml')

# CNN classifier
mouth_class = face_classify('model/model_CNN_mouth')
eye_class = face_classify('model/model_CNN_eyes')

# color
color_red = (0, 0, 255)
color_blue = (255, 0, 0)
color_white = (255, 255, 255)
color_black = (0, 0, 0)
state = ['Close', 'Open']
cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ---------- get the camera video size ---------------
    height, width = image.shape[:2]
    cv2.rectangle(image, (30, height - 30), (450, height - 30), color_black, 54)
    cv2.rectangle(image, (30, height - 30), (450, height - 30), color_white, 50)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=2)

    eyes = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=2)

    mouth = mouthCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=2)

    if faces.__len__() > 0:
        face_data = [faces[0]]
    else:
        face_data = [[0, 0, 0, 0]]

    eye_data = []
    for (x, y, w, h) in eyes:
        if x > face_data[0][0] and (x + w) < (face_data[0][0] + face_data[0][2]):           # out of x direction
            if y > face_data[0][1] and (y + h) < (face_data[0][1] + face_data[0][3]):       # out of y direction
                if (y + h/2) < (face_data[0][1] + face_data[0][3]/2):                       # below face
                    eye_data.append([x, y, w, h])

    mouth_data = []
    for (x, y, w, h) in mouth:
        if x > face_data[0][0] and x + w < face_data[0][0] + face_data[0][2]:               # out of x direction
            if y + h/2 < face_data[0][1] + face_data[0][3]:                                 # out of y direction
                if y > face_data[0][1] + face_data[0][3]/2:
                    if y < face_data[0][1] + face_data[0][3]:
                        if y + h > face_data[0][1] + face_data[0][3] * 0.8:
                            mouth_data.append([x, y, w, h])

    """ ------------------- get mouth image data for deep learning --------------- """
    ret_mouth = 1
    for (x, y, w, h) in mouth_data:
        img_mouth = gray[y:y + h, x:x + w]
        ret_mouth = mouth_class.classify(img_mouth)

    ret_eye = 0
    for (x, y, w, h) in eye_data:
        img_eye = gray[y:y + h, x:x + w]
        if eye_class.classify(img_eye):
            ret_eye = 1

    cv2.putText(image, "eye: " + state[ret_eye] + ", mouth: " + state[ret_mouth], (20, height - 20), cv2.FONT_HERSHEY_DUPLEX, 1, color_red, 2)

    # # Save the individual parts
    # if save_image == 1:
    #     t += 1
    #     eye = 0
    #     if t % 20 == 0:
    #         ind += 1
    #         print "save"
    #         for (x, y, w, h) in eye_data:
    #             img_eye = image[y:y + h, x:x + w]
    #             if eye == 0:
    #                 cv2.imwrite('eye1_' + ind.__str__() + '.bmp', img_eye)
    #                 eye = 1
    #             else:
    #                 cv2.imwrite('eye2_' + ind.__str__() + '.bmp', img_eye)
    #
    #         for (x, y, w, h) in mouth_data:
    #             img_mouth = image[y:y + h, x:x + w]
    #             cv2.imwrite('mouth_' + ind.__str__() + '.bmp', img_mouth)
    #
    #         for (x, y, w, h) in face_data:
    #             img_face = image[y:y + h, x:x + w]
    #             cv2.imwrite('face_' + ind.__str__() + '.bmp', img_face)

    # Draw a rectangle around the faces
    for (x, y, w, h) in eye_data:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y, w, h) in mouth_data:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    for (x, y, w, h) in face_data:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Faces found", image)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
