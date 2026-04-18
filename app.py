from flask import Flask, render_template, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import json
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and classes
model_path = 'models/flower_model.h5'
labels_path = 'models/class_labels.json'

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    # Compile the model to ensure it's ready for predictions
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model loaded successfully.")
else:
    print("Model not found! Run train.py first.")
    model = None

# Load class labels
classes = []
if os.path.exists(labels_path):
    with open(labels_path, 'r') as f:
        class_data = json.load(f)
        classes = class_data['classes']
        print(f"Loaded {len(classes)} classes from class_labels.json")
else:
    # Fallback to default classes
    classes = ['daisy', 'rose', 'sunflower', 'tulip', 'dandelion']
    print("Using default classes")

IMG_HEIGHT, IMG_WIDTH = 224, 224

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_image_bytes(image_bytes):
    """Preprocess image from bytes (webcam)"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_top_predictions(predictions, top_k=5):
    """Get top K predictions with confidence"""
    pred_probs = predictions[0]
    top_indices = np.argsort(pred_probs)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'class': classes[idx],
            'confidence': float(pred_probs[idx] * 100),
            'class_index': int(idx)
        })
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Single image prediction endpoint"""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img_array = preprocess_image(filepath)
        if img_array is None:
            return jsonify({'error': 'Failed to process image'}), 500
        
        predictions = model.predict(img_array, verbose=0)
        top_predictions = get_top_predictions(predictions)
        
        return jsonify({
            'prediction': top_predictions[0]['class'],
            'confidence': top_predictions[0]['confidence'],
            'top_predictions': top_predictions,
            'filename': filename,
            'timestamp': datetime.now().isoformat()
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict-webcam', methods=['POST'])
def predict_webcam():
    """Webcam image prediction endpoint"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image data'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image data'}), 400
    
    try:
        image_bytes = file.read()
        img_array = preprocess_image_bytes(image_bytes)
        
        if img_array is None:
            return jsonify({'error': 'Failed to process image'}), 500
        
        predictions = model.predict(img_array, verbose=0)
        top_predictions = get_top_predictions(predictions)
        
        return jsonify({
            'prediction': top_predictions[0]['class'],
            'confidence': top_predictions[0]['confidence'],
            'top_predictions': top_predictions,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint for multiple images"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    if not files or len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400
    
    results = []
    errors = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                img_array = preprocess_image(filepath)
                if img_array is None:
                    errors.append({'filename': filename, 'error': 'Failed to process'})
                    continue
                
                predictions = model.predict(img_array, verbose=0)
                top_predictions = get_top_predictions(predictions)
                
                results.append({
                    'filename': filename,
                    'prediction': top_predictions[0]['class'],
                    'confidence': top_predictions[0]['confidence'],
                    'top_predictions': top_predictions
                })
            except Exception as e:
                errors.append({'filename': file.filename, 'error': str(e)})
        else:
            errors.append({'filename': file.filename, 'error': 'Invalid file type'})
    
    return jsonify({
        'results': results,
        'errors': errors,
        'total_processed': len(results),
        'total_errors': len(errors),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'num_classes': len(classes),
        'classes': classes[:10] + ['...'] if len(classes) > 10 else classes
    })

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'webp'}

if __name__ == '__main__':
    print(f"Starting Flower Classification App with {len(classes)} classes")
    app.run(debug=True, port=5000)

