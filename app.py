import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# MobileNetV2 ImageNet Model
def mobilenetv2_imagenet():
    st.title("Image Classification with MobileNetV2")

    upload_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if upload_file is not None:
        image = Image.open(upload_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write("Classifiying...")

        # Load MobileNetV2 model
        mobil = tf.keras.applications.MobileNetV2(weights='imagenet')

        # Preprocess the image
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # Make predictions
        predictions = mobil.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]

        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{label}: {score * 100:.2f}%")


# CIFAR-10 Model
def cifar10_classification():
    st.title("CIFAR-10 Image Classification")

    Uploaded_File = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if Uploaded_File is not None:
        image = Image.open(Uploaded_File)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write("Classifying...")

        # Load CIFAR-10 model
        model = tf.keras.models.load_model("model111.h5")

        # CIFAR-10 class names
        class_names = ['aeroplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # Preprocess the image
        img = image.resize((32, 32))
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # Display the predicted class
        st.write(f"Predicted Class: {class_names[predicted_class]}")
        st.write(f"Confidence: {confidence * 100:.2f}%")


# Main Function
def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Select a Model", ["CIFAR-10 Image Classification", "MobileNetV2 Image Classification"])

    if choice == "MobileNetV2 Image Classification":
        mobilenetv2_imagenet()
    elif choice == "CIFAR-10 Image Classification":
        cifar10_classification()

if __name__ == "__main__":
    main()