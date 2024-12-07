from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import requests  

app = FastAPI()

# Inisialisasi model
MODEL_PATH = "models/mobilenet_model.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Mendapatkan detail input dan output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Label klasifikasi (urutannya harus sesuai dengan model)
LABELS = ["healthy", "rust", "scab"]

def preprocess_image(image_path, target_size):
    """Membaca dan memproses gambar untuk model TFLite."""
    img = Image.open(image_path).convert("RGB")  
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalisasi
    return np.expand_dims(img_array, axis=0).astype(np.float32)  # Tambahkan batch dimension


LARAVEL_API_BASE_URL = "http://127.0.0.1:8000/api"  # Ganti dengan URL Laravel

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Simpan file upload sementara
        temp_path = f"data/{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        # Preprocessing gambar
        input_data = preprocess_image(temp_path, target_size=(224, 224))
        os.remove(temp_path)  # Hapus file sementara

        # Prediksi menggunakan model TFLite
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]  # Ambil hasil prediksi

        # Ambil label dengan probabilitas tertinggi
        predicted_index = np.argmax(predictions)
        predicted_label = LABELS[predicted_index]

        # Request ke Laravel untuk mengambil data kategori
        category_response = requests.get(f"{LARAVEL_API_BASE_URL}/categories/{predicted_label}")
        if category_response.status_code != 200:
            return JSONResponse(content={"predicted_label": predicted_label, "error": "Category not found"})

        category_data = category_response.json()

        # Return hasil prediksi + data kategori
        return JSONResponse(content={
            "predicted_label": predicted_label,
            "category": category_data
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)