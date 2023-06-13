from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import tensorflow as tf
from PIL import Image

app=Flask(__name__)

class_names=['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy']

model=load_model('latest_model.h5')


def prediction(image_path):
    img = Image.open(image_path)
    img=img.resize((256,256))
    img_to_tensor = tf.convert_to_tensor(img)
    reshaped_tensor = tf.expand_dims(img_to_tensor, axis=0)
    # first_image=img_to_tensor.numpy().astype('uint8')
    # print(first_image.shape)
    # first_image=first_image/255.0
    # print(first_image)
    # first_image=[first_image]
    # print(first_image.shape)
    # first_image=tf.convert_to_tensor(first_image)
    # print(first_image.shape)
    batch_prediction=model.predict(reshaped_tensor)
    print(batch_prediction)
    return class_names[np.argmax(batch_prediction)]


@app.route('/',methods=['GET'])
def mainpage():
    return render_template('index.html')


@app.route('/',methods=['POST'])
def submit():
    image_file=request.files['image_file']
    image_path="static/"+image_file.filename
    image_file.save(image_path)
    
    predicted=prediction(image_path)
    print(predicted)

    return render_template('index.html', predicted=predicted, image_path=image_path)


if __name__=="__main__":
    app.run(debug=True)