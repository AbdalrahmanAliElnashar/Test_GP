from flask import Flask, request, render_template, url_for, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
from keras.utils import load_img, img_to_array


app = Flask(__name__)

# def preprossing(image):
#     image = load_img(image, target_size=(150, 150))
#     image_array =  img_to_array(image)
#     image_array = image_array.reshape(1, 150, 150, 3)
#     return image_array

def preprossing(image):
    image=Image.open(image)
    image = image.resize((150, 150))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 150, 150, 3)
    return image_arr

classes = ['Allergy', 'Bacteria']
model=load_model('Augmentation_weights_model.h5')

@app.route('/')
def index():

    return render_template('index.html', appName="Skin Diseases Classification")


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        print("image loaded....") 
        image_arr= preprossing(image)
        print("predicting ...")
        new_predict = model.predict(image_arr)
        new_predict = np.round(new_predict).flatten().astype('int32')
        Class = new_predict[0]
        #print(Class)

        classes = {
            0: 'Allergy', 
            1: 'Bacteria'
        }

        # print(classes[Class])
        prediction = classes[Class]
        # print("Model predicting ...")
        # result = model.predict(image_arr)
        # print("Model predicted")
        # Class = np.round(result).astype('int32')
        # prediction = classes[Class]
        # print(prediction)
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        img = request.files['fileup']
        print("image loaded....")
        image_arr= preprossing(img)
        print("predicting ...")
        new_predict = model.predict(image_arr)
        new_predict = np.round(new_predict).flatten().astype('int32')
        Class = new_predict[0]
        #print(Class)

        classes = {
            0: 'Allergy', 
            1: 'Bacteria'
        }

        # print(classes[Class])
        prediction = classes[Class]

        return render_template('index.html', prediction=prediction, appName="Skin Diseases Classification")
    else:
        return render_template('index.html',appName="Skin Diseases Classification")


if __name__ == '__main__':
    app.run(debug=True)