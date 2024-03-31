# from flask import Flask, request, render_template
# import cv2
# import numpy as np

# app = Flask(__name__)

# # Load your trained model
# model = ...  # Load your trained model here

# # Function to preprocess image
# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_resized = cv2.resize(img, (224, 224))
#     img_reshaped = np.reshape(img_resized, (1, 224, 224, 3))
#     return img_reshaped

# # Function to predict image
# def predict_image(image_path):
#     img = preprocess_image(image_path)
#     predictions = model.predict(img)
#     threshold = 0.5
#     predicted_label = "Tumpukan Sampah" if predictions > threshold else "Bukan Tumpukan Sampah"
#     return predicted_label

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             file.save('uploads/' + file.filename)
#             image_path = 'uploads/' + file.filename
#             predicted_label = predict_image(image_path)
#             return render_template('result.html', image_path=image_path, predicted_label=predicted_label)
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model  # Tambahkan ini untuk impor model H5

# Inisialisasi Flask
app = Flask(__name__)

# Load model deteksi sampah
model = load_model('model.h5')  # Ganti 'model.h5' dengan path ke model Anda

@app.route('/')  
def index():
    return render_template('index.html')

# Pastikan dekorator route diberi indentasi dengan benar
@app.route('/upload', methods=['POST'])  
def upload():
    # Di sini, kode di dalam fungsi upload() harus diindentasi dengan benar
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return 'File uploaded successfully'
    return 'No file uploaded'

@app.route('/predict', methods=['POST'])
def predict():
    # Terima file gambar yang diunggah oleh pengguna
    file = request.files['file']
    if file:
        # Baca file gambar
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        # Lakukan prediksi menggunakan model
        # ...
        # Di sini tambahkan kode untuk melakukan prediksi menggunakan model
        # ...
        # Misalnya:
        predicted_label = "Sampah" if model.predict(img) else "Bukan Sampah"
        # Simpan hasil prediksi ke dalam file HTML
        save_prediction_to_html(img, predicted_label)
        return jsonify({'result': predicted_label})
    else:
        return jsonify({'error': 'No file uploaded'})

def save_prediction_to_html(image, prediction):
    # Simpan gambar dan hasil prediksi ke dalam file HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">

    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hasil Prediksi</title>
    </head>

    <body>
        <h1>Hasil Prediksi</h1>
        <img src="data:image/png;base64,{image_to_base64(image)}" alt="Gambar">
        <p>Label Prediksi: {prediction}</p>
    </body>

    </html>
    """
    with open('result.html', 'w') as file:
        file.write(html_content)

def image_to_base64(image):
    # Konversi gambar ke format base64 untuk disisipkan ke dalam HTML
    retval, buffer = cv2.imencode('.png', image)
    base64_image = base64.b64encode(buffer)
    return base64_image.decode()

if __name__ == '__main__':
    app.run(debug=True)
