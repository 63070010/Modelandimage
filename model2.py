import tensorflow as tf
from tensorflow.keras.models import Model
import cv2

from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

categories = ['2017 Hatchback', '2018 Yaris ATIV',
              '2019 Yaris ATIV', '2019 Yaris Sedan', '2020 Hatchback']

# load model

model = tf.keras.models.load_model('D:/ModelImg/Modelandimage/myCarsModel.h5')
print(model.summary())


def preprareImage(PathForImage):
    image = load_img(PathForImage, target_size=(224, 224))
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult, axis=0)
    imgResult = imgResult / 255.
    return imgResult


testImage = "D:/ModelImg/Modelandimage/maxresdefault.jpg"

imgForModel = preprareImage(testImage)
resultArray = model.predict(imgForModel, verbose=1)
answer = np.argmax(resultArray, axis=1)

print(answer)

index = answer[0]

print("The predicted car is : " + categories[index])

# show the image :

img = cv2.imread(testImage)
cv2.putText(img, categories[index], (10, 100),
            cv2.FONT_HERSHEY_COMPLEX, 1.6, (255, 0, 0), 3, cv2.LINE_AA)
cv2.imshow('image', img)
cv2.waitKey(0)
