import streamlit as st
import tensorflow as tf
import os


class_names = ["COVID-19", "Normal", "Pneumonia"]


@st.cache(suppress_st_warning=True)

def preprocess_image(path):
    img = tf.keras.preprocessing.image.load_img(path , grayscale=False, color_mode='rgb', target_size=(224,224,3), interpolation='nearest')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array*(1./255)
    img_array = tf.expand_dims(img_array, 0)
    return img, img_array

def efnet(img_array):
    tflite_interpreter = tf.lite.Interpreter(model_path="EfNet_B4_model_final.tflite")
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    tflite_interpreter.allocate_tensors()
    tflite_interpreter.set_tensor(input_details[0]["index"], img_array)
    tflite_interpreter.invoke()
    tflite_model_prediction = tflite_interpreter.get_tensor(output_details[0]["index"])
    tflite_model_prediction = tflite_model_prediction.squeeze().argmax(axis = 0)
    print(tflite_model_prediction)
    pred_class = class_names[tflite_model_prediction]
    print(pred_class)
    print("This X-Ray image most likely belongs to {} .".format(class_names[tflite_model_prediction]))
    #inference = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    #print(inference)
    return "This X-Ray image most likely belongs to {} .".format(class_names[tflite_model_prediction])


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center; color: white;'>Chest X-Ray Image Classification</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Image of X-Ray.....", type= ["jpeg", "png", "jpg"])

    if uploaded_file is not None:
        with open(os.path.join("tempDir/",uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())

        path = os.path.join("tempDir/",uploaded_file.name)
        img, img_array = preprocess_image(path)
        st.image(img)

    if st.button('Get Prediction'):
        st.success(efnet(img_array))
        
        
