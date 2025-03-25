import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import cv2
from PIL import Image

# load the models when import "predictions.py"
model_elbow_frac = tf.keras.models.load_model("weights/ResNet50_Elbow_frac.h5")
model_hand_frac = tf.keras.models.load_model("weights/ResNet50_Hand_frac.h5")
model_shoulder_frac = tf.keras.models.load_model("weights/ResNet50_Shoulder_frac.h5")
model_parts = tf.keras.models.load_model("weights/ResNet50_BodyParts.h5")

# categories for each result by index

#   0-Elbow     1-Hand      2-Shoulder
categories_parts = ["Elbow", "Hand", "Shoulder"]

#   0-fractured     1-normal
categories_fracture = ['fractured', 'normal']

def is_xray_image(img_path):
    """Check if the image appears to be an X-ray"""
    try:
        # Load image and convert to grayscale
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)
        
        # Calculate image statistics
        mean_val = np.mean(img_array)
        std_val = np.std(img_array)
        
        # Calculate histogram
        hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
        hist = hist / img_array.size  # Normalize
        
        # Check for X-ray characteristics:
        # 1. Mostly dark with bright areas (low to medium mean)
        # 2. High contrast (high std)
        # 3. Significant peaks in histogram
        dark_peak = np.max(hist[:50]) > 0.01  # At least 1% pixels in dark range
        bright_peak = np.max(hist[200:]) > 0.01  # At least 1% pixels in bright range
        
        is_xray = (
            (30 < mean_val < 180) and  # Not too dark or too bright
            (std_val > 40) and         # Good contrast
            (dark_peak or bright_peak)  # Has significant peaks
        )
        
        # Special case for very dark but high-contrast X-rays
        if mean_val < 50 and std_val > 70:
            is_xray = True
            
        return is_xray
    except Exception as e:
        print(f"Error checking X-ray: {e}")
        return False

# get image and model name, the default model is "Parts"
# Parts - bone type predict model of 3 classes
# otherwise - fracture predict for each part
def predict(img, model="Parts"):
    # First verify it's an X-ray image
    if not is_xray_image(img):
        return "not_xray"
    
    size = 224
    if model == 'Parts':
        chosen_model = model_parts
    else:
        if model == 'Elbow':
            chosen_model = model_elbow_frac
        elif model == 'Hand':
            chosen_model = model_hand_frac
        elif model == 'Shoulder':
            chosen_model = model_shoulder_frac

    # load image with 224px224p (the training model image size, rgb)
    temp_img = image.load_img(img, target_size=(size, size))
    x = image.img_to_array(temp_img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    prediction = np.argmax(chosen_model.predict(images), axis=1)

    # chose the category and get the string prediction
    if model == 'Parts':
        prediction_str = categories_parts[prediction.item()]
    else:
        prediction_str = categories_fracture[prediction.item()]

    return prediction_str
