from main import Classifier
import cv2
from PIL import Image

classifier = Classifier()
classifier.load_model()

image = Image.open('/content/drive/MyDrive/xongxoa/Emotion-ResNet50/Data-Fer2013/train/1068632.jpg')
cv2.show(image)
classifier.predict(image)