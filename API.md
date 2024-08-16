# AI Object Detection Framework API Documentation

## Introduction

This API allows you to integrate the object detection capabilities of the AI Object Detection Framework into your applications. It provides endpoints to detect objects in images and videos, and supports automatic model selection based on confidence intervals.

## Model Selection Process

The backend uses a decision tree to select the most appropriate YOLOv10 model based on the confidence intervals of detected objects.

1. **Set a minimum confidence threshold.**
2. **Evaluate detections using each model size:**
   - If all detections meet the minimum confidence, select that model size.
   - If no models meet the confidence threshold, the largest model (`x`) is used.

## Endpoints

### POST /detect
- **Description:** Detect objects in an image.
- **Request Parameters:**
  - `file`: The image file to upload.
  - `auto_select`: `true` for automatic model selection, `false` for manual model selection.
  - `min_confidence` (optional): Minimum confidence for automatic model selection (default is 0.9).
  - `model_size` (optional): Model size for manual model selection (`n`, `s`, `m`, `b`, `l`, `x`).
  - `target_language` (optional): Language code for object names translation (default is 'en').
- **Response:** Annotated image.

### POST /get_detections
- **Description:** Get detection data for an image.
- **Request Parameters:**
  - `file`: The image file to upload.
  - `auto_select`: `true` for automatic model selection, `false` for manual model selection.
  - `min_confidence` (optional): Minimum confidence for automatic model selection (default is 0.9).
  - `model_size` (optional): Model size for manual model selection (`n`, `s`, `m`, `b`, `l`, `x`).
  - `target_language` (optional): Language code for object names translation (default is 'en').
- **Response:** JSON object containing detection data with translated names.

### POST /detect_video
- **Description:** Detect objects in a video.
- **Request Parameters:**
  - `file`: The video file to upload.
  - `auto_select`: `true` for automatic model selection, `false` for manual model selection.
  - `min_confidence` (optional): Minimum confidence for automatic model selection (default is 0.9).
  - `model_size` (optional): Model size for manual model selection (`n`, `s`, `m`, `b`, `l`, `x`).
  - `target_language` (optional): Language code for object names translation (default is 'en').
- **Response:** Annotated video.

## Error Handling

The API returns standard HTTP status codes to indicate the success or failure of a request. Common status codes include:

- **200 OK:** The request was successful.
- **400 Bad Request:** There was an issue with the request parameters.
- **500 Internal Server Error:** An error occurred on the server.

For more detailed error messages, check the response body.

## Example Usage

Here is an example of how to make a request to the `/detect` endpoint using `curl`:

```bash
curl -X POST "http://your-server-url/detect" \
-H "Content-Type: multipart/form-data" \
-F "file=@/path/to/your/image.jpg" \
-F "auto_select=true" \
-F "min_confidence=0.9" \
-F "target_language=en"
```

