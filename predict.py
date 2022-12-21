from main import Classifier
import cv2
from PIL import Image
import tensorflow as tf
import align.detect_face
import numpy as np
import facenet

INPUT_IMAGE_SIZE = 96
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
frame = cv2.imread("im2.jpg", cv2.IMREAD_COLOR)
tf.compat.v1.disable_eager_execution()
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
with sess.as_default():
    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "align")
    bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

    faces_found = bounding_boxes.shape[0]
    try:
        if faces_found > 1:
            cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (255, 255, 255), thickness=1, lineType=2)
        elif faces_found > 0:
            det = bounding_boxes[:, 0:4]
            bb = np.zeros((faces_found, 4), dtype=np.int32)
            for i in range(faces_found):
                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]
                print(bb[i][3]-bb[i][1])
                print(frame.shape[0])
                print((bb[i][3]-bb[i][1])/frame.shape[0])
                if (bb[i][3]-bb[i][1])/frame.shape[0]>0.25:
                    cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                    scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                        interpolation=cv2.INTER_CUBIC)
                    scaled = facenet.prewhiten(scaled)
    except:
       pass
classifier = Classifier()
classifier.load_model()


 
img = scaled.copy()

classifier.predict(img)
cv2.imshow("image", img)