import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("TensorFlow-Streamlit-main\saved_model\save_model.h5")
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

map_dict = {0: 'object 23',
            1:  'Object 24',
            2:  'Object 25',
            3:  'Object 26',
            4:  'Object 27',
            5:  'Object 28',
            6:  'Object 29',
            7:  'Object 30',
            8:  'Object 31',
            9:  'Object 32',
            10:  'Object 33',
            11:  'Object 34',
            12: 'Object 35',
            13: 'Object 36',
            14: 'Object 46',
            15:  'Object 47',
            16: 'Object 48',
            17: 'Object 49',
            18: 'Object 50',
            19: 'Object 51',
            20: 'Object 52',
            21: 'Object 53',
            22: 'Object 54',
            23: 'Object 55',
            24: 'Object 56',
            25: 'Object 57',
            26: 'Object 58',
            27: 'Object 59',
            28: 'Object 60',
            29: 'Object 61',
            30: 'Object 62',
            31: 'Object 63',
            32: 'Object 64',
            33: 'Object 65',
            34: 'Object 66',
            35: 'Object 67',
            36: 'Object 68',
            37: 'Object 69',
            38: 'Object 70',
            39: 'Object 71',
            40: 'Object 72'}


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))