from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import io
from PIL import Image
import datetime

app = Flask(__name__)
CORS(app)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('skin_lesion.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, classification TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

# Mock pre-trained model (simplified for demo)
def create_mock_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')  # 2 classes: benign, malignant
    ])
    # Load pre-trained weights (mocked here; use real weights from HAM10000 training)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Preprocess image
def preprocess_image(image):
    img = image.resize((64, 64))  # Resize to match model input
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Store prediction in database
def store_prediction(classification, confidence):
    conn = sqlite3.connect('skin_lesion.db')
    c = conn.cursor()
    c.execute('INSERT INTO predictions (timestamp, classification, confidence) VALUES (?, ?, ?)',
              (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), classification, confidence))
    conn.commit()
    conn.close()

# Mock classification (simulate HAM10000-like behavior)
model = create_mock_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Unsupported file format'}), 400

        img = Image.open(io.BytesIO(file.read()))
        img_array = preprocess_image(img)
        prediction = model.predict(img_array)
        confidence = prediction[0][1] * 100  # Probability of malignant
        classification = 'Malignant' if confidence > 50 else 'Benign'

        store_prediction(classification, confidence)
        return jsonify({
            'classification': classification,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
