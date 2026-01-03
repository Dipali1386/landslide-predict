import os
import pandas as pd
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib  



csv_path = "C:/Users/dipali/OneDrive/Desktop/ratings.csv"
image_folder = "C:/Users/dipali/OneDrive/Desktop/images"


df = pd.read_csv(csv_path)
df.columns = ['Name', 'Label']


def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (100, 100))
    return img.flatten() / 255.0  


features = []
labels = []

for idx, row in df.iterrows():
    fname = str(row["Name"]).strip()

    if fname.isdigit():
        fname = f"{int(fname):03}.jpg"

    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        print(f" Skipping invalid filename: {fname}")
        continue

    path = os.path.join(image_folder, fname)
    feats = extract_features(path)

    if feats is not None:
        features.append(feats)
        labels.append(row["Label"])
    else:
        print(f" Could not read image: {path}")


X = np.array(features)
y = np.array(labels)


if len(X) == 0:
    print(" No valid images found. Check filenames or folder path.")
    exit()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train) 


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f"\n Model trained successfully! Mean Squared Error: {mse:.4f}\n")


print("Sample predictions:")
for actual, pred in zip(y_test[:10], y_pred[:10]):
    print(f"Actual: {actual:.2f} | Predicted: {pred:.2f}")


plt.figure(figsize=(8, 5))


plt.scatter(y_test, y_pred, c='blue', label='Predicted', zorder=2)


plt.scatter(y_test, y_test, c='green', marker='x', label='Actual', zorder=3)


for actual, pred in zip(y_test, y_pred):
    plt.plot([actual, actual], [actual, pred], 'gray', linewidth=0.5, zorder=1)


plt.plot([0, 1], [0, 1], 'r--', label='Perfect prediction', zorder=0)

plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Landslide Risk: Predicted vs Actual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


model_file = "C:/Users/dipali/OneDrive/Desktop/landslide_model.pkl"
joblib.dump(model, model_file)
print(f"\n Model saved as: {model_file}")