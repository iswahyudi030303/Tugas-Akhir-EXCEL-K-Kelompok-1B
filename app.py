from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Ganti dengan kunci rahasia yang aman

# Konfigurasi database SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Model database untuk menyimpan history
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(150), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Kamus kelas
dic = {0: 'Cardiomegaly', 1: 'Normal', 2: 'Pneumonia', 3: 'Pneumothorax'}

# Construct the absolute path to the model file
model_path = os.path.join(os.getcwd(), 'ParuMobileNet2-Paru2 MobileNet2-99.68.h5')

# Muat model
model = load_model(model_path)

# Fungsi untuk memproses gambar input
def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalisasi jika diperlukan oleh model
    return img_array

def predict_label(img_path):
    # Praproses gambar
    img_array = preprocess_image(img_path)
    
    # Prediksi
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    return dic[pred_class]

@app.route("/", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['authenticated'] = True
        return redirect(url_for('index'))
    return render_template("login.html")

@app.route("/index")
def index():
    if not session.get('authenticated'):
        return redirect(url_for('login'))
    return render_template("index.html")

@app.route("/main", methods=['GET', 'POST'])
def main():
    if not session.get('authenticated'):
        return redirect(url_for('login'))
    return render_template("classification.html")

@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)
        
        # Simpan hasil prediksi ke database
        history_entry = History(image_path=img_path, prediction=p)
        db.session.add(history_entry)
        db.session.commit()
        
        return render_template("classification.html", prediction=p, img_path=img_path)

@app.route("/history")
def history():
    if not session.get('authenticated'):
        return redirect(url_for('login'))
    histories = History.query.all()
    return render_template("history.html", histories=histories)

@app.route("/delete/<int:id>")
def delete(id):
    if not session.get('authenticated'):
        return redirect(url_for('login'))
    history_entry = History.query.get_or_404(id)
    try:
        db.session.delete(history_entry)
        db.session.commit()
        return redirect(url_for('history'))
    except:
        return 'There was a problem deleting that entry'

@app.route("/deskripsi", methods=['GET'])
def deskripsi():
    return render_template("deskripsi.html")
    
@app.route("/tips")
def tips():
    return render_template("tips.html")

@app.route("/help")
def help():
    return render_template("help.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

# Tambahan routes untuk halaman spesifik
@app.route("/pneumonia")
def pneumonia():
    return render_template("pneumonia.html")

@app.route("/pneumotorax")
def pneumotorax():
    return render_template("pneumotorax.html")

@app.route("/cardiomegaly")
def cardiomegaly():
    return render_template("cardiomegaly.html")

if __name__ == '__main__':
    # Buat database jika belum ada
    with app.app_context():
        db.create_all()
    app.run(debug=True)
