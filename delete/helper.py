import cv2
import numpy as np

def preprocessing(img):
    img = np.array(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resized_img = cv2.resize(gray_img,(48,48))
    print(resized_img.shape)
    resized_img = resized_img.reshape((1,48,48,1))
    print(resized_img.shape)
    return resized_img