import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model("traffic_sign_classifier.keras")

class_names = {
    0: '20 km/h', 1: '30 km/h', 2: '50 km/h', 3: '60 km/h', 4: '70 km/h',
    5: '80 km/h', 6: 'End of 80 km/h', 7: '100 km/h', 8: '120 km/h', 9: 'No passing',
    10: 'No passing > 3.5t', 11: 'Right-of-way next int.', 12: 'Priority road',
    13: 'Yield', 14: 'Stop', 15: 'No vehicles', 16: 'Vehicles > 3.5t prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve left',
    20: 'Dangerous curve right', 21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End all speed/pass limits', 33: 'Turn right ahead', 34: 'Turn left ahead',
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left',
    38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory',
    41: 'End of no passing', 42: 'End of no passing > 3.5t'
}

def classify_traffic_sign(input_image):
    img = Image.fromarray(input_image).convert('RGB').resize((32, 32))
    img_array = np.array(img, dtype='float32') / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)

    preds = model.predict(input_tensor, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1][:3]
    return {class_names[idx]: float(preds[idx]) for idx in top_indices}

demo = gr.Interface(
    fn=classify_traffic_sign,
    inputs=gr.Image(type="numpy", label="Upload Road Sign"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="Traffic Sign Recognition System",
    description="Upload a photo of a road sign to predict its class using deep learning."
)

if __name__ == "__main__":
    demo.launch()
