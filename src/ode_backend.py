"""
Module summary: Object Detection using YOLOv10 model.

Author: Faycal Kilali
Credits: yolov10

Citation:
- YOLOv10: Real-Time End-to-End Object Detection
  Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, Guiguang Ding
  arXiv preprint arXiv:2405.14458, 2024
"""

from flask import Flask, request, jsonify, send_file
import torch
import os
import wget
import numpy as np
from io import BytesIO
from PIL import Image
import tempfile
import cv2
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)

from ultralytics import YOLOv10

# List of model sizes in ascending order
model_sizes = ['n', 's', 'm', 'b', 'l', 'x']
model_urls = {size: f'https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{size}.pt' for size in model_sizes}
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

# Function to detect objects in an image and return annotated image
def detect_objects(image_path, model_size='n'):
    model = models[model_size]
    print(f"Using model size: {model_size} for detection in: {image_path}")
    results = model(image_path)  # This line should work with model API

    if not results or len(results) == 0:
        raise FileNotFoundError(f"No results returned from model for image path: {image_path}")

    annotated_image = results[0].plot()
    detections = []
    for box in results[0].boxes:
        detection = {
            'name': results[0].names[int(box.cls)],  # Get class name using class index
            'confidence': float(box.conf),
            'box': box.xyxy.tolist()
        }
        detections.append(detection)

    # Convert annotated image from numpy array to PIL Image
    if isinstance(annotated_image, np.ndarray):
        annotated_image_pil = Image.fromarray(annotated_image)
    else:
        raise TypeError("Annotated image is not a numpy array")

    return annotated_image_pil, detections

def get_best_model(image_path, min_confidence):
    for size in model_sizes:
        _, detections = detect_objects(image_path, model_size=size)
        if all(d['confidence'] >= min_confidence for d in detections):
            return size, detections
    return model_sizes[-1], detections  # Return the largest model if none meet the threshold

# Function to detect objects in a video and return annotated video
def detect_objects_in_video(video_path, model_size='n'):
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

        results = model(frame)  # Ensure model API is correct for processing video frames
        if not results or len(results) == 0:
            continue  # Skip frames without results

        annotated_frame = results[0].plot()
        if isinstance(annotated_frame, np.ndarray):
            out.write(annotated_frame)
        else:
            raise TypeError("Annotated frame is not a numpy array")

    cap.release()
    out.release()

    return temp_output_file.name

# Function to get the best model size for video
def get_best_model_for_video(video_path, min_confidence):
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
        _, detections = detect_objects(temp_image_path, model_size=size)
        if all(d['confidence'] >= min_confidence for d in detections):
            return size

    return model_sizes[-1]  # Return the largest model if none meet the threshold

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

    # Log the file path
    print(f"File saved to: {file_path}")

    # Schedule file deletion after 60 minutes
    delete_file_after_timeout(file_path, 60)

    auto_select = request.form.get('auto_select') == 'true'
    try:
        if auto_select:
            min_confidence = float(request.form.get('min_confidence', DEFAULT_MINIMUM_INFERENCE))
            best_model_size, detections = get_best_model(file_path, min_confidence)
        else:
            model_size = request.form.get('model_size', 'n')
            best_model_size, detections = detect_objects(file_path, model_size=model_size)

        # Create annotated image
        annotated_image_pil, detections = detect_objects(file_path, model_size=best_model_size)

        # Save annotated image to temporary file
        temp_annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'annotated_{filename}')
        annotated_image_pil.save(temp_annotated_image_path, 'JPEG')

        # Log the temp file path
        print(f"Annotated image saved to: {temp_annotated_image_path}")

        # Schedule annotated image deletion
        delete_file_after_timeout(temp_annotated_image_path, 60)

        response = send_file(temp_annotated_image_path, mimetype='image/jpeg')
        response.headers['Model-Size'] = best_model_size
        return response
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {str(e)}")  # Log the specific file error
        return jsonify({'error': f'FileNotFoundError: {str(e)}'}), 500
    except Exception as e:
        print(f"Error in /detect: {str(e)}")  # Log general errors
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
    try:
        if auto_select:
            min_confidence = float(request.form.get('min_confidence', DEFAULT_MINIMUM_INFERENCE))
            best_model_size, detections = get_best_model(file_path, min_confidence)
        else:
            model_size = request.form.get('model_size', 'n')
            best_model_size, detections = detect_objects(file_path, model_size=model_size)

        response_data = {
            'model_size': best_model_size,
            'detections': detections
        }
        return jsonify(response_data)
    except Exception as e:
        print(f"Error in /get_detections: {str(e)}")  # Log the error
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

    # Schedule file deletion after 60 minutes
    delete_file_after_timeout(file_path, 60)

    auto_select = request.form.get('auto_select') == 'true'
    try:
        if auto_select:
            min_confidence = float(request.form.get('min_confidence', DEFAULT_MINIMUM_INFERENCE))
            best_model_size = get_best_model_for_video(file_path, min_confidence)
        else:
            best_model_size = request.form.get('model_size', 'n')

        annotated_video_path = detect_objects_in_video(file_path, model_size=best_model_size)

        # Schedule annotated video deletion
        delete_file_after_timeout(annotated_video_path, 60)

        return send_file(annotated_video_path, mimetype='video/mp4')
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {str(e)}")  # Log the specific file error
        return jsonify({'error': f'FileNotFoundError: {str(e)}'}), 500
    except Exception as e:
        print(f"Error in /detect_video: {str(e)}")  # Log general errors
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def file_too_large(e):
    return "File is too large, max file size is 16 MB.", 413

if __name__ == '__main__':
    app.run(debug=True)
