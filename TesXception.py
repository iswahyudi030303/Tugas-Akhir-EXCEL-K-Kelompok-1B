import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Path ke model dan gambar
model_path = 'Deteksi Paru Exception\Paru2-Batik Exception-98.66.h5'
image_path = 'Dataset Fix\\Pneumothorax\\4_train_1_.png'
#Dataset Fix\Pneumothorax\4_train_1_.png
#F:\STECHOQ\Chest X-Rays.v4i.folder\test\PNEUMONIA\person22_bacteria_77_jpeg.rf.cce10b1b32782999d6a69f0e347f01ba.jpg
# Muat model
model = load_model(model_path)

# Fungsi untuk praproses gambar
def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalisasi jika diperlukan oleh model
    return img_array

# Praproses gambar
img_array = preprocess_image(image_path)

# Prediksi
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)
classes = ['Normal', 'Pneumonia', 'Pneumothorax']  # Sesuaikan dengan kelas yang Anda miliki

# Tampilkan hasil
print(f"Prediksi: {classes[predicted_class[0]]}, Confidence: {np.max(prediction)}")