import streamlit as st
from PIL import Image
import requests
import io

# URL of the Flask back-end
backend_url = 'http://127.0.0.1:5000'

# Streamlit app
st.title('Object Detection and Translation')

st.markdown(
    """
    ### Disclaimer
    This application uses the YOLOv10 model for object detection. Please do not upload any sensitive or personal images. The application is not intended for commercial use. The maximum upload size is 16 MB.
    """
)

st.sidebar.markdown(
    """
    ### Credits
    - Faycal Kilali
    - YOLOv10: Real-Time End-to-End Object Detection by Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, Guiguang Ding. ArXiv preprint arXiv:2405.14458, 2024
    """
)

uploaded_file = st.file_uploader("Upload an image or video file", type=["jpg", "jpeg", "png", "mp4"])

# Language selection for translation
target_language = st.sidebar.selectbox(
    "Select language for object names translation:",
    ('en', 'es', 'fr', 'de', 'zh-cn')  # Add more languages as needed
)

if uploaded_file is not None:
    file_type = uploaded_file.type
    filename = uploaded_file.name  # Preserve the original filename
    if file_type.startswith("image"):
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        st.write("")
        st.write("Detecting objects...")

        auto_select = st.radio(
            "Model selection mode:",
            ('Automatic', 'Manual')
        )

        if auto_select == 'Manual':
            model_size = st.selectbox(
                "Choose model size:",
                ('n', 's', 'm', 'b', 'l', 'x')
            )
            min_confidence = None
        else:
            min_confidence = st.slider(
                "Select minimum confidence for automatic model selection:",
                0.0, 1.0, 0.9
            )
            model_size = None

        if st.button('Detect Objects'):
            with st.spinner('Processing...'):
                image_bytes = io.BytesIO()
                image.save(image_bytes, format='JPEG')
                image_bytes.seek(0)

                response = requests.post(
                    f'{backend_url}/detect',
                    files={'file': (filename, image_bytes, 'image/jpeg')},
                    data={
                        'auto_select': 'true' if auto_select == 'Automatic' else 'false',
                        'min_confidence': min_confidence if auto_select == 'Automatic' else None,
                        'model_size': model_size if auto_select == 'Manual' else None,
                        'target_language': target_language
                    }
                )

                if response.status_code == 200:
                    try:
                        annotated_image_bytes = io.BytesIO(response.content)
                        annotated_image = Image.open(annotated_image_bytes)
                        st.image(annotated_image, caption='Annotated Image.', use_column_width=True)

                        model_size = response.headers.get('Model-Size', 'unknown')
                        st.write(f"Model used: YOLOv10{model_size}")

                        # Temporary annotated file name fix
                        annotated_filename = f'annotated_{filename}'

                        detections_response = requests.post(
                            f'{backend_url}/get_detections',
                            files={'file': (annotated_filename, image_bytes, 'image/jpeg')},
                            data={
                                'auto_select': 'true' if auto_select == 'Automatic' else 'false',
                                'min_confidence': min_confidence if auto_select == 'Automatic' else None,
                                'model_size': model_size if auto_select == 'Manual' else None,
                                'target_language': target_language
                            }
                        )

                        # Debugging information
                        response_content = detections_response.content
                        st.write(f"Raw response content: {response_content}")

                        if detections_response.status_code == 200:
                            if response_content:
                                detections_data = detections_response.json()
                                detections = detections_data['detections']
                                model_size = detections_data['model_size']
                                st.write(f"Model used for detections: YOLOv10{model_size}")

                                st.write("Detections:")
                                for detection in detections:
                                    st.write(
                                        f"{detection['translated_name']} - Confidence: {detection['confidence']:.2f} - Box: {detection['box']}"
                                    )

                                st.download_button(
                                    'Download Annotated Image',
                                    data=annotated_image_bytes.getvalue(),
                                    file_name='annotated_image.jpg',
                                    mime='image/jpeg'
                                )
                            else:
                                st.write("No detections data received.")
                        else:
                            st.write(
                                f"Error in detections. Status code: {detections_response.status_code}, Response content: {response_content.decode()}")
                    except ValueError as e:
                        st.write(f"Error parsing detections response: {e}")
                else:
                    st.write(
                        f"Error in object detection. Status code: {response.status_code}, Response content: {response.text}")

    elif file_type.startswith("video"):
        st.video(uploaded_file)

        st.write("")
        st.write("Detecting objects in video...")

        auto_select = st.radio(
            "Model selection mode:",
            ('Automatic', 'Manual')
        )

        if auto_select == 'Manual':
            model_size = st.selectbox(
                "Choose model size:",
                ('n', 's', 'm', 'b', 'l', 'x')
            )
            min_confidence = None
        else:
            min_confidence = st.slider(
                "Select minimum confidence for automatic model selection:",
                0.0, 1.0, 0.9
            )
            model_size = None

        if st.button('Detect Objects in Video'):
            with st.spinner('Processing...'):
                video_bytes = uploaded_file.read()

                response = requests.post(
                    f'{backend_url}/detect_video',
                    files={'file': (filename, video_bytes, 'video/mp4')},
                    data={
                        'auto_select': 'true' if auto_select == 'Automatic' else 'false',
                        'min_confidence': min_confidence if auto_select == 'Automatic' else None,
                        'model_size': model_size if auto_select == 'Manual' else None,
                        'target_language': target_language
                    }
                )

                if response.status_code == 200:
                    st.video(io.BytesIO(response.content))
                    st.download_button(
                        'Download Annotated Video',
                        data=response.content,
                        file_name='annotated_video.mp4',
                        mime='video/mp4'
                    )
                else:
                    st.write("Error in video object detection.")
else:
    st.info("Please upload an image or video.")

# API Documentation
st.markdown(
    """
    ## API Documentation

    You can use this API to integrate object detection into your applications. The API provides endpoints to detect objects in images and videos.

    ### Endpoints

    - **POST /detect**
      - **Description:** Detect objects in an image.
      - **Request Parameters:**
        - `file`: The image file to upload.
        - `auto_select`: `true` for automatic model selection, `false` for manual model selection.
        - `min_confidence` (optional): Minimum confidence for automatic model selection (default is 0.9).
        - `model_size` (optional): Model size for manual model selection (`n`, `s`, `m`, `b`, `l`, `x`).
        - `target_language` (optional): Language code for object names translation (default is 'en').
      - **Response:** Annotated image.

    - **POST /get_detections**
      - **Description:** Get detection data for an image.
      - **Request Parameters:**
        - `file`: The image file to upload.
        - `auto_select`: `true` for automatic model selection, `false` for manual model selection.
        - `min_confidence` (optional): Minimum confidence for automatic model selection (default is 0.9).
        - `model_size` (optional): Model size for manual model selection (`n`, `s`, `m`, `b`, `l`, `x`).
        - `target_language` (optional): Language code for object names translation (default is 'en').
      - **Response:** JSON object containing detection data with translated names.

    - **POST /detect_video**
      - **Description:** Detect objects in a video.
      - **Request Parameters:**
        - `file`: The video file to upload.
        - `auto_select`: `true` for automatic model selection, `false` for manual model selection.
        - `min_confidence` (optional): Minimum confidence for automatic model selection (default is 0.9).
        - `model_size` (optional): Model size for manual model selection (`n`, `s`, `m`, `b`, `l`, `x`).
        - `target_language` (optional): Language code for object names translation (default is 'en').
      - **Response:** Annotated video.
    """
)
