import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.applications import imagenet_utils


filename= "dog.jpg"

"""
Several method to load image in the workbook
1.with the help of IPython library
    from IPthon.display import Image
    Image(filename="",width =224, height)

2.with the help of keras api(need to use matplotlib to display the image)
    from tensorflow.keras.preprocessing import image
    img= image.load_img(filename, target_size=(224,224)

3.with the help of open cv  
    img=cv2.imread(filename)
    plt.imshow(img)

4.from PIL import Image ##pip install Pillow
    img= Image.open(filename)
    img= img.resize((224,224))
    plt.imshow(img) 
"""

img= image.load_img(filename, target_size=(224,224))
#plt.imshow(img)
#plt.show()


#LOADING THE DEEP LEARNIING MODEL

mobile1= tf.keras.applications.mobilenet.MobileNet() ##first version

from tensorflow.keras.preprocessing import image
img= image.load_img(filename, target_size=(224,224))

#plt.imshow(img)
#plt.show()

resized_img= image.img_to_array(img)
final_image=np.expand_dims(resized_img,axis=0) ##need of forth dimension
final_image=tf.keras.applications.mobilenet.preprocess_input(final_image)

print(resized_img.shape)

print(final_image.shape)

predictions=mobile1.predict(final_image)

results1= imagenet_utils.decode_predictions(predictions)

print(results1) ##results of first version


mobile2= tf.keras.applications.mobilenet_v2.MobileNetV2() ##Version 2

predictions=mobile2.predict(final_image)

results2= imagenet_utils.decode_predictions(predictions)

print(results2) ##results of version 2
