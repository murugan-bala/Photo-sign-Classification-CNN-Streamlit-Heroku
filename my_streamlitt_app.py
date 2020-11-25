# install streamlit : pip install streamlit
# upgrade protobuf:  pip3 install --upgrade protobuf
# To run the code : streamlit run rps_app.py

import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from keras.preprocessing import image

model = load_model('sign_classifier_1.h5')
IMG_SHAPE = 150

st.title("""
          Photo or Signature classification Prediction
         """
         )
st.subheader("This is a simple image classification web app to predict Face or signature ")
file = st.file_uploader("Please upload an image file", type=["jpg", "png","jpeg"])
#print(file)

def import_and_predict(image_data, model):
        print("Inside predict")
        img_array = np.array(image_data)
        cv2.imwrite('out.jpg',cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))  # C:\\SIFY\\SIBI\\Sign_classification\\
        #test_image = image.load_img('C:\\Users\\Murugan\\Desktop\\class_test\\a2.jpg', target_size = (IMG_SHAPE, IMG_SHAPE))
        test_image = image.load_img('out.jpg', target_size = (IMG_SHAPE, IMG_SHAPE))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        #training_set.class_indices
        print(result)
        print()
        
        return result
if file is None:
    st.text("Please upload an image file")
else:
    imagee = Image.open(file)
    #image = file.read()
    st.image(imagee, use_column_width=True)
    result = import_and_predict(imagee, model)
    if result[0][0] == 1:
        prediction = 'photo'
        print(prediction)
        st.write("It is a photo!")
    else:
        prediction = 'signature'
        print(prediction)
        st.markdown('**It is a signature!**.')
        #st.markdown('Streamlit is **_really_ cool**.')
        #st.write("It is a signature!")
    #st.text("Probability (0: Signature, 1: Photo )")
    st.write("Predicted sucessfully...")
