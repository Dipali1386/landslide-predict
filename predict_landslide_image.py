import cv2
import numpy as np
import joblib


model_path = "C:/Users/dipali/Desktop/landslide_model.pkl"
model = joblib.load(model_path)


image_path = "C:/Users/dipali/Desktop/img9.png"



def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(" Error: Cannot read image at:", image_path)
        exit()
    img = cv2.resize(img, (100, 100))
    return img.flatten() / 255.0

features = extract_features(image_path).reshape(1, -1)

prediction = model.predict(features)[0]
print(f"\n Predicted Landslide Risk Grading: {prediction:.2f} (scale 0 to 1)")