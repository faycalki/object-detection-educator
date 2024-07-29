"""
Module summary: Object Detection using YOLOv10 model.

Author: Faycal Kilali
Credits: yolov10

Citation:
- YOLOv10: Real-Time End-to-End Object Detection
  Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, Guiguang Ding
  arXiv preprint arXiv:2405.14458, 2024
"""

import os
import tempfile
import threading

import cv2
import numpy as np
import wget
from PIL import Image
from deep_translator import GoogleTranslator
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)

from ultralytics import YOLOv10

# List of model sizes in ascending order
model_sizes = ['n', 's', 'm', 'b', 'l', 'x']
model_urls = {size: f'https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{size}.pt' for size in
              model_sizes}
models = {}

# Ensure the necessary directories exist
UPLOAD_FOLDER = './uploads'
MODEL_FOLDER = './models'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

# Ensure the models are downloaded and loaded
for size in model_sizes:
    model_path = os.path.join(MODEL_FOLDER, f'yolov10{size}.pt')
    if not os.path.exists(model_path):
        print(f"Downloading model {model_path}...")
        wget.download(model_urls[size], model_path)
    models[size] = YOLOv10(model_path)

DEFAULT_MINIMUM_INFERENCE = 0.9
# Maximum file size configuration for FLASK
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit for uploads
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def translate_name(name, target_language):
    try:
        translated_text = GoogleTranslator(source='en', target=target_language).translate(name)
        print(f"Translated '{name}' as: '{translated_text}'")
        return translated_text
    except Exception as e:
        print(f"Translation error: {e}")
        return name  # Return original name if translation fails


def detect_objects(image_path, model_size='n', target_language='en'):
    model = models[model_size]
    print(f"Using model size: {model_size} for detection in: {image_path}")
    results = model(image_path)

    if not results or len(results) == 0:
        raise FileNotFoundError(f"No results returned from model for image path: {image_path}")

    annotated_image = results[0].plot()
    detections = []
    for box in results[0].boxes:
        detection = {
            'name': results[0].names[int(box.cls)],
            'confidence': float(box.conf),
            'box': box.xyxy.tolist(),
            'translated_name': translate_name(results[0].names[int(box.cls)], target_language)  # Add translated name
        }
        detections.append(detection)

    if isinstance(annotated_image, np.ndarray):
        annotated_image_pil = Image.fromarray(annotated_image)
    else:
        raise TypeError("Annotated image is not a numpy array")

    return annotated_image_pil, detections


def choose_model_based_on_confidence(detections, min_confidence):
    """
    Choose the model based on the detections' confidence levels.
    """
    for size in model_sizes:
        if all(d['confidence'] >= min_confidence for d in detections):
            return size
    return model_sizes[-1]

def get_best_model(image_path, min_confidence, target_language='en'):
    for size in model_sizes:
        _, detections = detect_objects(image_path, model_size=size, target_language=target_language)
        if all(d['confidence'] >= min_confidence for d in detections):
            return size, detections
    return model_sizes[-1], detections


def detect_objects_in_video(video_path, model_size='n', target_language='en'):
    model = models[model_size]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_output_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        if not results or len(results) == 0:
            continue

        annotated_frame = results[0].plot()
        if isinstance(annotated_frame, np.ndarray):
            out.write(annotated_frame)
        else:
            raise TypeError("Annotated frame is not a numpy array")

    cap.release()
    out.release()

    return temp_output_file.name


def get_best_model_for_video(video_path, min_confidence, target_language='en'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise FileNotFoundError(f"Could not read frame from video path: {video_path}")

    temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
    cv2.imwrite(temp_image_path, frame)

    for size in model_sizes:
        _, detections = detect_objects(temp_image_path, model_size=size, target_language=target_language)
        if all(d['confidence'] >= min_confidence for d in detections):
            return size

    return model_sizes[-1]

def delete_file_after_timeout(file_path, timeout):
    def delete_file():
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path} after {timeout} minutes.")

    threading.Timer(timeout * 60, delete_file).start()

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({'error': 'Invalid file name'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    print(f"File saved to: {file_path}")

    delete_file_after_timeout(file_path, 60)

    auto_select = request.form.get('auto_select') == 'true'
    target_language = request.form.get('target_language', 'en')  # Retrieve target language

    try:
        if auto_select:
            min_confidence = float(request.form.get('min_confidence', DEFAULT_MINIMUM_INFERENCE))
            best_model_size, detections = get_best_model(file_path, min_confidence, target_language)
        else:
            model_size = request.form.get('model_size', 'n')
            best_model_size, detections = detect_objects(file_path, model_size=model_size, target_language=target_language)

        annotated_image_pil, detections = detect_objects(file_path, model_size=best_model_size, target_language=target_language)

        temp_annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'annotated_{filename}')
        annotated_image_pil.save(temp_annotated_image_path, 'JPEG')

        print(f"Annotated image saved to: {temp_annotated_image_path}")

        delete_file_after_timeout(temp_annotated_image_path, 60)

        response = send_file(temp_annotated_image_path, mimetype='image/jpeg')
        response.headers['Model-Size'] = best_model_size
        return response
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {str(e)}")
        return jsonify({'error': f'FileNotFoundError: {str(e)}'}), 500
    except Exception as e:
        print(f"Error in /detect: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_detections', methods=['POST'])
def get_detections():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({'error': 'Invalid file name'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    auto_select = request.form.get('auto_select') == 'true'
    target_language = request.form.get('target_language', 'en')  # Retrieve target language

    try:
        if auto_select:
            min_confidence = float(request.form.get('min_confidence', DEFAULT_MINIMUM_INFERENCE))
            best_model_size, detections = get_best_model(file_path, min_confidence, target_language)
        else:
            model_size = request.form.get('model_size', 'n')
            best_model_size, detections = detect_objects(file_path, model_size=model_size, target_language=target_language)

        response_data = {
            'model_size': best_model_size,
            'detections': detections
        }
        return jsonify(response_data)
    except Exception as e:
        print(f"Error in /get_detections: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect_video', methods=['POST'])
def detect_video():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({'error': 'Invalid file name'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    delete_file_after_timeout(file_path, 60)

    auto_select = request.form.get('auto_select') == 'true'
    target_language = request.form.get('target_language', 'en')  # Retrieve target language

    try:
        if auto_select:
            min_confidence = float(request.form.get('min_confidence', DEFAULT_MINIMUM_INFERENCE))
            best_model_size = get_best_model_for_video(file_path, min_confidence, target_language)
        else:
            best_model_size = request.form.get('model_size', 'n')

        annotated_video_path = detect_objects_in_video(file_path, model_size=best_model_size, target_language=target_language)

        delete_file_after_timeout(annotated_video_path, 60)

        return send_file(annotated_video_path, mimetype='video/mp4')
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {str(e)}")
        return jsonify({'error': f'FileNotFoundError: {str(e)}'}), 500
    except Exception as e:
        print(f"Error in /detect_video: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def file_too_large(e):
    return "File is too large, max file size is 16 MB.", 413

if __name__ == '__main__':
    app.run(debug=True)
