from keras.models import load_model
from PIL import Image
import numpy as np

from keras.utils import load_img, img_to_array




model=load_model('Augmentation_weights_model.h5')

def preprossing(image):
    image = load_img(image, target_size=(150, 150))
    image_array =  img_to_array(image)
    image_array = image_array.reshape(1, 150, 150, 3)
    return image_array



print("image loaded....")
image_arr= preprossing('allergy.jpg')
print("predicting ...")
new_predict = model.predict(image_arr)
new_predict = np.round(new_predict).flatten().astype('int32')
Class = new_predict[0]
print(Class)

classes = {
    0: 'Allergy', 
    1: 'Bacteria'
}

print(classes[Class])
# result = model.predict(image_arr)
# print("predicted ...")
# Class = np.round(result).flatten().astype('int32')
# prediction = classes[Class]

# print(prediction)
