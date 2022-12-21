from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import glob
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
from imgaug import augmenters as iaa
import numpy as np
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

class Classifier:
    def __init__(self):
        self.model = None
        self.list_labels = ['Angry', 'Disgust','Fear','Happy','Neutral','Sad','Surprise']
        self.label_map = {item: idx for idx, item in enumerate(self.list_labels)}

    def build_model(self):
        input_layer = Input(shape=[96, 96, 3])
        preprocess_layer = preprocess_input(input_layer)

        backbone = ResNet50(input_shape=[96, 96, 3], include_top=False, weights = 'imagenet')
        backbone_output_layer = backbone(preprocess_layer)

        for layer in backbone.layers[:-3]:
          layer.trainable = False

        flatten_layer = Flatten()(backbone_output_layer)
        output_layer = Dense(10, activation='softmax')(flatten_layer)

        model = Model(input_layer, output_layer)
        model.summary()

        loss = SparseCategoricalCrossentropy()
        optimizer = Adam(learning_rate=0.0001)
        metric = SparseCategoricalAccuracy()
        
        model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
        self.model = model

    def load_model(self):
      self.model = load_model('models/emotion.h5')

    def save_model(self):
        pass

    def train(self, train_path, valid_path):
        train_generator = DataLoader(64, train_path, self.label_map)
        valid_generator = DataLoader(64, train_path, self.label_map)

        tensorboard = TensorBoard(log_dir="/content/drive/MyDrive/xongxoa/Emotion-ResNet50/Graph")

        call_backs = [
            EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10),
            ModelCheckpoint('models/emotion.h5',monitor = 'val_loss',save_best_only = True, verbose = 1),
            TensorBoard(log_dir='/content/drive/MyDrive/xongxoa/Emotion-ResNet50/Graph', histogram_freq=0, write_graph=True, write_images=True)
        ]
        self.model.fit(train_generator, validation_data = valid_generator, epochs = 200, callbacks=[call_backs, tensorboard])

    def predict(self, image):
        """
        :param image: a PIL image with arbitrary size
        :return: a string ('Angry', 'Disgust','Fear','Happy','Neutral','Sad','Surprise')
        """
        np_image = np.array(image.resize([96, 96]))
        batch_images = np.array([np_image])
        y_prob_batch = self.model.predict(batch_images)
        y_pred_batch = np.argmax(y_prob_batch, axis = 1)
        y_predict = y_pred_batch[0]
        print("Kết quả mô hình dự đoán là:")
        
        return self.list_labels[y_predict]