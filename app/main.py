from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
import numpy as np
import os

app = FastAPI()

# Inisialisasi model
MODEL_PATH = "models/mobilenet_model.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Mendapatkan detail input dan output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_path, target_size):
    """Membaca dan memproses gambar untuk model TFLite."""
    img = Image.open(image_path).resize(target_size)
    img_array = np.array(img) / 255.0  # Normalisasi
    return np.expand_dims(img_array, axis=0).astype(np.float32)

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
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # Return hasil prediksi
        return JSONResponse(content={
            "predictions": predictions.tolist()
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
