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


@app.route('/supported_languages', methods=['GET', 'POST'])
def supported_languages():
    """
    Returns a list of supported languages for translation.

    This function is a Flask route that handles a POST request to the '/supported_languages' endpoint.
    It returns a JSON response containing the supported languages that the GoogleTranslator library
    can translate object names into.

    Args:
        None

    Returns:
        JSON: A JSON object containing the list of supported languages.

    Example:
        curl -X POST http://localhost:5000/supported_languages
    """
    try:
        # Get the list of supported languages from GoogleTranslator
        languages = GoogleTranslator().get_supported_languages(as_dict=False)  # output: {arabic: ar, french: fr, english:en etc...}

        response_data = {
            'supported_languages': languages
        }
        return jsonify(response_data)
    except Exception as e:
        print(f"Error in /supported_languages: {str(e)}")
        return jsonify({'error': str(e)}), 500

def translate_name(name, target_language, source_language):
    """
    Translates a given name from the Source Language to the specified target language using the Deep-Learning Language class.

    Args:
        source_language: The source language to translate from
        name (str): The name to be translated.
        target_language (str): The target language code to translate the name into.

    Returns:
        str: The translated name if successful, otherwise the original name.

    Raises:
        Exception: If there is an error during the translation process.
    """
    try:
        translated_text = GoogleTranslator(source=source_language, target=target_language).translate(name)
        return translated_text
    except Exception as e:
        print(f"Translation error: {e}")
        return name  # Return original name if translation fails


def detect_objects(image_path, model_size='n', target_language='en', source_language='en'):
    """
        Detects objects in an image using a specified model size and translates the object names to the target language.

        Args:
            source_language: The source language to translate from
            image_path (str): The path to the image file.
            model_size (str, optional): The size of the model to use for detection. Defaults to 'n'.
            target_language (str, optional): The target language code to translate the object names into. Defaults to 'en'.

        Returns:
            Tuple[PIL.Image.Image, List[Dict[str, Union[str, float, List[int]]]]]: A tuple containing the annotated image as a PIL Image object and a list of dictionaries representing the detected objects. Each dictionary contains the following keys:
                - 'name' (str): The name of the detected object.
                - 'confidence' (float): The confidence score of the detection.
                - 'box' (List[int]): The bounding box coordinates of the detected object.
                - 'translated_name' (str): The translated name of the detected object.

        Raises:
            FileNotFoundError: If no results are returned from the model for the given image path.
            TypeError: If the annotated image is not a numpy array.

    """
    model = models[model_size]
    print(f"Using model size: {model_size} for detection in: {image_path}")
    results = model(image_path)

    if not results or len(results) == 0:
        raise FileNotFoundError(f"No results returned from model for image path: {image_path}")

    annotated_image = results[0].plot()
    detections = []
    for box in results[0].boxes:
        detection = {
            'name': translate_name(results[0].names[int(box.cls)], source_language='en', target_language=source_language), # Translate from English to the actual source language, as we don't have a hardcoded list of all the objects in every language available.
            'confidence': float(box.conf),
            'box': box.xyxy.tolist(),
            'translated_name': translate_name(name=results[0].names[int(box.cls)], target_language=target_language, source_language=source_language)  # Add translated name of the object
        }
        print(detection['name'])
        detections.append(detection)

    if isinstance(annotated_image, np.ndarray):
        annotated_image_pil = Image.fromarray(annotated_image)
    else:
        raise TypeError("Annotated image is not a numpy array")

    return annotated_image_pil, detections


def choose_model_based_on_confidence(detections, min_confidence):
    """
        Choose the model based on the confidence levels of the detections.

        Args:
            detections (List[Dict[str, Union[str, float, List[int]]]]): A list of dictionaries representing the detected objects. Each dictionary contains the following keys:
                - 'name' (str): The name of the detected object.
                - 'confidence' (float): The confidence score of the detection.
                - 'box' (List[int]): The bounding box coordinates of the detected object.
                - 'translated_name' (str): The translated name of the detected object.
            min_confidence (float): The minimum confidence score to consider a detection as valid.

        Returns:
            str: The model size that meets the minimum confidence requirement among the provided detections.

    """
    for size in model_sizes:
        if all(d['confidence'] >= min_confidence for d in detections):
            return size
    return model_sizes[-1]


def get_best_model(image_path, min_confidence, target_language='en', source_language='en'):
    """
    Get the best model for the given image path based on the confidence levels of the detections.

    Args:
        image_path (str): The path to the image.
        min_confidence (float): The minimum confidence score to consider a detection as valid.
        target_language (str, optional): The target language for translation. Defaults to 'en'.

    Returns:
        Tuple[str, List[Dict[str, Union[str, float, List[int]]]]]: A tuple containing the best model size and a list of dictionaries representing the detected objects. Each dictionary contains the following keys:
            - 'name' (str): The name of the detected object.
            - 'confidence' (float): The confidence score of the detection.
            - 'box' (List[int]): The bounding box coordinates of the detected object.
            - 'translated_name' (str): The translated name of the detected object.

            If no model meets the minimum confidence requirement, the last model size in the `model_sizes` list is returned along with the detections.
    """
    for size in model_sizes:
        _, detections = detect_objects(image_path, model_size=size, target_language=target_language, source_language=source_language)
        if all(d['confidence'] >= min_confidence for d in detections):
            return size, detections
    return model_sizes[-1], detections


def detect_objects_in_video(video_path, model_size='n', target_language='en'):
    """
    Detects objects in a video using a specified YOLOv10 model size.
    Args:
        video_path (str): The path to the video file.
        model_size (str, optional): The size of the YOLOv10 model to use. Defaults to 'n'.
        target_language (str, optional): The target language for translating object names. Defaults to 'en'.
    Returns:
        str: The path to the temporary video file with annotated frames.
    Raises:
        FileNotFoundError: If the video file cannot be opened.
    Note:
        This function uses the YOLOv10 model to detect objects in the video frames. It writes the annotated frames
        to a temporary video file and returns its path. The annotated frames include bounding boxes and labels
        for the detected objects. The target_language parameter is used for translating the object names to the
        specified language.
    Example:
        >>> detect_objects_in_video('path/to/video.mp4', 'm', 'fr')
        'path/to/temp_video.mp4'
    """
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
    """
        Get the best model size for a given video file based on the confidence levels of the detected objects.

        Args:
            video_path (str): The path to the video file.
            min_confidence (float): The minimum confidence score to consider a detection as valid.
            target_language (str, optional): The target language for translating object names. Defaults to 'en'.

        Returns:
            str: The model size that meets the minimum confidence requirement among the detected objects in the first frame of the video. If no detection meets the requirement, the last model size in the list is returned.

        Raises:
            FileNotFoundError: If the video file cannot be opened or if the first frame of the video cannot be read.

    """
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
    """
        Schedules the deletion of a file after a specified timeout.

        Args:
            file_path (str): The path to the file to be deleted.
            timeout (int): The duration in minutes after which the file will be deleted.

        Returns:
            None

        This function creates a timer that, after the specified timeout, deletes the file
        located at the given file path. If the file does not exist at the time of deletion,
        no action is taken. The function does not return anything.

        Example:
            delete_file_after_timeout('/path/to/file.txt', 10)
            # The file at '/path/to/file.txt' will be deleted after 10 minutes.
    """

    def delete_file():
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path} after {timeout} minutes.")

    threading.Timer(timeout * 60, delete_file).start()


@app.route('/detect', methods=['POST'])
def detect():
    """
    Detect objects in an uploaded image file and return the annotated image.

    This function is a Flask route that handles a POST request to the '/detect' endpoint.
    It expects a file to be uploaded in the request. If no file is uploaded, it returns a JSON
    response with an error message and a status code of 400. If the filename is invalid, it
    returns a JSON response with an error message and a status code of 400.

    The uploaded file is saved to a temporary folder specified in the Flask application's
    configuration. After a specified timeout, the file is deleted.

    The function also retrieves the values of the 'auto_select' and 'target_language' form
    fields. If 'auto_select' is set to 'true', the function calls the 'get_best_model' function
    to determine the best model size for object detection based on the uploaded file and the
    target language. Otherwise, it calls the 'detect_objects' function with the specified model
    size and target language.

    The function then calls the 'detect_objects' function again with the best model size and
    target language to obtain the annotated image and the detections. The annotated image is
    saved to a temporary file in the same folder. After a specified timeout, the file is deleted.

    The function returns a response with the annotated image and the model size as a header.

    If a 'FileNotFoundError' occurs during the execution of the function, it returns a JSON
    response with an error message and a status code of 500. If any other exception occurs, it
    returns a JSON response with the error message and a status code of 500.

    Args:
        None

    Returns:
        A Flask response object with the annotated image and the model size as a header.

    Raises:
        FileNotFoundError: If the uploaded file or the annotated image file is not found.
        Exception: If any other error occurs during the execution of the function.

    Example:
        curl -X POST -F "file=@/path/to/image.jpg" -F "auto_select=true" -F "target_language=en" -F "source_language=en"
        http://localhost:5000/detect

    """
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
    source_language = request.form.get('source_language', 'en') # Retrieve Source Language

    try:
        if auto_select:
            min_confidence = float(request.form.get('min_confidence', DEFAULT_MINIMUM_INFERENCE))
            best_model_size, detections = get_best_model(file_path, min_confidence, target_language, source_language)
        else:
            model_size = request.form.get('model_size', 'n')
            best_model_size, detections = detect_objects(file_path, model_size=model_size,
                                                         target_language=target_language, source_language=source_language)

        annotated_image_pil, detections = detect_objects(file_path, model_size=best_model_size,
                                                         target_language=target_language, source_language=source_language)

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
    """
        Retrieves detections from an uploaded file.

        This function is an endpoint for the '/get_detections' route. It receives a POST request with a file
        parameter named 'file'. The function checks if the file is present and valid. If not, it returns a
        JSON response with an error message and a 400 status code.

        If the file is valid, the function retrieves the filename, checks if it is valid, and constructs the
        file path. It then retrieves the 'auto_select' and 'target_language' parameters from the request
        form. If 'auto_select' is true, it retrieves the 'min_confidence' parameter as well.

        The function calls either the 'get_best_model' or 'detect_objects' function, depending on the
        'auto_select' parameter, to determine the 'best_model_size' and 'detections'. It constructs a
        response data dictionary with the 'model_size' and 'detections' keys.

        If an exception occurs during the process, the function catches it, prints an error message, and
        returns a JSON response with the error message and a 500 status code.

        Parameters:
            None

        Returns:
            A JSON response with the 'model_size' and 'detections' keys, or a JSON response with an error
            message and a 500 status code.
    """
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({'error': 'Invalid file name'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    auto_select = request.form.get('auto_select') == 'true'
    source_language = request.form.get('source_language', 'en')
    target_language = request.form.get('target_language', 'en')  # Retrieve target language

    try:
        if auto_select:
            min_confidence = float(request.form.get('min_confidence', DEFAULT_MINIMUM_INFERENCE))
            best_model_size, detections = get_best_model(file_path, min_confidence, target_language)
        else:
            model_size = request.form.get('model_size', 'n')
            best_model_size, detections = detect_objects(file_path, model_size=model_size,
                                                         target_language=target_language, source_language=source_language)

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
    """
        Detects objects in a video file and returns an annotated video file.

        Parameters:
            None

        Returns:
            A video file with annotated frames in the 'video/mp4' format, or a JSON response with an error
            message and a 500 status code.

        Raises:
            FileNotFoundError: If the video file cannot be opened.
            Exception: If an error occurs during the object detection process.

        Note:
            This function expects a POST request with a file parameter named 'file' containing the video file
            to be uploaded. The function saves the file to the 'UPLOAD_FOLDER' directory, detects objects
            in the video using the YOLOv10 model, and returns an annotated video file. The 'auto_select'
            parameter determines whether to automatically select the model size based on the minimum
            confidence, or to use a manually selected model size. The 'target_language' parameter is used
            for translating object names to the specified language.

        Example:
            POST /detect_video
            {
                "file": <video file>,
                "auto_select": true,
                "min_confidence": 0.9,
                "target_language": "en"
            }
            Response:
                <annotated video file>
    """
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

        annotated_video_path = detect_objects_in_video(file_path, model_size=best_model_size,
                                                       target_language=target_language)

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
    """
     Handles the case when a file is too large

     Args:
         e (Exception): The exception raised when the file is too large

     Returns:
         tuple: A tuple containing a string message and an integer status code.
                The message is "File is too large, max file size is 16 MB."
                The status code is 413.
    """
    return "File is too large, max file size is 16 MB.", 413


if __name__ == '__main__':
    app.run(debug=True)
