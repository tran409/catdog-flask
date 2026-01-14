import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("mo_hinh_cnn.h5")

# Class labels
class_names = ["Chó", "Mèo"]

# Hàm dự đoán
def predict(image):
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = image.reshape(1, 150, 150, 3)
    prediction = model.predict(image)[0]
    class_index = np.argmax(prediction)
    confidence = prediction[class_index]
    return {class_names[0]: float(prediction[0]), class_names[1]: float(prediction[1])}

# Giao diện Gradio
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Nhận diện Mèo và Chó",
    description="Tải ảnh lên để phân loại là mèo hay chó"
)

demo.launch()
