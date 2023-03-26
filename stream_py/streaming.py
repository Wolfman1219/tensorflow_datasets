import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
from  PIL import Image

option = st.selectbox(
    'Modelni tanlang',
    ('Cifar10', 'Cifar100', 'Fashion MNIST', 'MNIST'))

def upload_file():
    file = st.file_uploader(label="rasmni kiriting" )
    return file


def drawable():
    canvas_result_1 = st_canvas(
    # Fixed fill color with some opacity
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=20,
    stroke_color="#000000",
    background_color="#FFFFFF",
    update_streamlit=True,
    height=400,
    width=400,
    drawing_mode="freedraw",
    key="canvas1",
    )
    return canvas_result_1
    # get the prediction from your model and return itif canvas_result.image_data is not None and predict:


def image_resizer(img, model):
    if model.startswith("Cifar"):
        res = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
    else:
        res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)        
    return res



def get_prediction(image, model):
    image = image_resizer(img=image, model=model['name'])
    if model['name'].endswith("MNIST"):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,70,255,0)
        image = np.array([thresh])
    # else:
    #     pass
    else:
        image = np.array([image])
    return np.argmax(model['model'].predict(image))
    
    # pass 
    # get the prediction from your model and return itif canvas_result.image_data is not None and predict:
    # st.text("Prediction : {}".format(prediction))
# prediction = get_prediction(canvas_result.image_data)


def load_model(model):
    Model = tf.keras.Model()
    Model = tf.keras.models.load_model(f"stream_model/{model}.h5")
    return {"name":model, "model":Model}



if option.startswith("Cifar"):
    # file = upload_file()
    file = st.file_uploader(label="rasmni kiriting" )
    if file is not None:
        print(file)
        image = np.asarray(Image.open(file))
        
    # file = upload_file().read() 
    # image = cv2.imread(file.name)
        prediction_ = get_prediction(image, load_model(option))
        st.write(prediction_)

elif option.endswith("MNIST"):
    # st.image(drawable().image_data)
    # print(drawable().image_data)
    prediction_ = get_prediction(drawable().image_data, load_model(option))
    st.write(prediction_)
