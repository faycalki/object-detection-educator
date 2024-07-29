import graphviz
from graphviz import Digraph
import streamlit as st
from PIL import Image
import requests
import io
from io import BytesIO


def create_decision_tree():
    dot = Digraph()
    dot.attr(rankdir='TB', size='10,10', splines='ortho', nodesep='1', ranksep='1')

    # Root node
    dot.node('input', 'Input Stage', shape='rect')

    # Branches for Image and Video
    dot.node('image', 'Image', shape='ellipse')
    dot.node('video', 'Video', shape='ellipse')
    dot.edge('input', 'image')
    dot.edge('input', 'video')

    # Detection Process Nodes
    dot.node('img_detection', 'Detection Process', shape='rect')
    dot.node('vid_detection', 'Detection Process', shape='rect')
    dot.edge('image', 'img_detection')
    dot.edge('video', 'vid_detection')

    # Model Selection Mode
    dot.node('img_model_mode', 'Model Selection Mode', shape='rect')
    dot.node('vid_model_mode', 'Model Selection Mode', shape='rect')
    dot.edge('img_detection', 'img_model_mode')
    dot.edge('vid_detection', 'vid_model_mode')

    # Automatic vs Manual
    dot.node('img_auto', 'Automatic', shape='ellipse')
    dot.node('img_manual', 'Manual', shape='ellipse')
    dot.edge('img_model_mode', 'img_auto')
    dot.edge('img_model_mode', 'img_manual')

    dot.node('vid_auto', 'Automatic', shape='ellipse')
    dot.node('vid_manual', 'Manual', shape='ellipse')
    dot.edge('vid_model_mode', 'vid_auto')
    dot.edge('vid_model_mode', 'vid_manual')

    # Min Confidence Check for Automatic
    dot.node('img_conf_check', 'Min Confidence Check', shape='rect')
    dot.node('vid_conf_check', 'Min Confidence Check', shape='rect')
    dot.edge('img_auto', 'img_conf_check')
    dot.edge('vid_auto', 'vid_conf_check')

    # Model Size Selection for Manual
    dot.node('img_model_size', 'Model Size Selection', shape='rect')
    dot.node('vid_model_size', 'Model Size Selection', shape='rect')
    dot.edge('img_manual', 'img_model_size')
    dot.edge('vid_manual', 'vid_model_size')

    # Min Confidence Check Result
    dot.node('img_pass', 'Pass (≥)', shape='ellipse', color='green')
    dot.node('img_fail', 'Fail (<)', shape='ellipse', color='red')
    dot.edge('img_conf_check', 'img_pass', label='Yes')
    dot.edge('img_conf_check', 'img_fail', label='No')

    dot.node('vid_pass', 'Pass (≥)', shape='ellipse', color='green')
    dot.node('vid_fail', 'Fail (<)', shape='ellipse', color='red')
    dot.edge('vid_conf_check', 'vid_pass', label='Yes')
    dot.edge('vid_conf_check', 'vid_fail', label='No')

    # Largest Model (Default x) for Fail Case
    dot.node('img_largest_model', 'Use Largest Model (Default x)', shape='rect')
    dot.node('vid_largest_model', 'Use Largest Model (Default x)', shape='rect')
    dot.edge('img_fail', 'img_largest_model')
    dot.edge('vid_fail', 'vid_largest_model')

    # Final Detection Outcome
    dot.node('img_detection_outcome', 'Detection Outcome', shape='rect')
    dot.node('vid_detection_outcome', 'Detection Outcome', shape='rect')
    dot.edge('img_pass', 'img_detection_outcome', label='Use Model Size')
    dot.edge('img_largest_model', 'img_detection_outcome', label='Use Largest Model')
    dot.edge('vid_pass', 'vid_detection_outcome', label='Use Model Size')
    dot.edge('vid_largest_model', 'vid_detection_outcome', label='Use Largest Model')

    return dot


def create_single_model_tree(model_size, detections, min_confidence):
    dot = Digraph()
    dot.attr(rankdir='TB', size='10,10', ratio='fill', splines='ortho', nodesep='1', ranksep='1')

    dot.node('root', f'YOLOv10{model_size}', shape='rect', width='2.5', height='1.0', fontsize='12')

    for j, detection in enumerate(detections):
        detection_name = detection.get('translated_name', f'Detection_{j}')
        detection_node = f'detection_{j}'
        label = f'{detection_name}: Pass' if detection['confidence'] >= min_confidence else f'{detection_name}: Fail'
        color = 'green' if detection['confidence'] >= min_confidence else 'red'
        dot.edge('root', detection_node, label=label, color=color, fontsize='10')

        dot.node(detection_node, f'{detection_name}\n(Confidence: {detection["confidence"]:.2f})', shape='ellipse', fontsize='10')

    return dot


def visualize_decision_tree(detections, min_confidence, model_size):
    st.write("### Decision Tree Visualization")
    st.write(f"Decision-making process based on confidence intervals for the model YOLOv10{model_size}:")

    model_sizes = ['n', 's', 'm', 'b', 'l', 'x']
    if model_size not in model_sizes:
        st.error(f"Model size {model_size} is not valid.")
        return

    dot = create_single_model_tree(model_size, detections, min_confidence)

    tree_image = dot.pipe(format='png')
    st.image(BytesIO(tree_image), caption=f'Decision Tree for YOLOv10{model_size}', use_column_width=True)

backend_url = 'http://127.0.0.1:5000'

st.title('Object Detection and Translation')

st.markdown("""
    ### Disclaimer
    This application uses the YOLOv10 model for object detection. Please do not upload any sensitive or personal images. The application is not intended for commercial use. The maximum upload size is 16 MB.
""")

st.sidebar.markdown("""
    ### Credits
    - Faycal Kilali
    - YOLOv10: Real-Time End-to-End Object Detection by Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, Guiguang Ding. ArXiv preprint arXiv:2405.14458, 2024
""")

# Display the decision tree
st.write("### Decision Tree for Detection Procedure")
decision_tree = create_decision_tree()
tree_image = decision_tree.pipe(format='png')
st.image(tree_image, caption='Decision Tree for Detection Procedure', use_column_width=True)


uploaded_file = st.file_uploader("Upload an image or video file", type=["jpg", "jpeg", "png", "mp4"])

target_language = st.sidebar.selectbox(
    "Select language for object names translation:",
    ('en', 'es', 'fr', 'de', 'zh-cn')
)

if uploaded_file is not None:
    file_type = uploaded_file.type
    filename = uploaded_file.name
    if file_type.startswith("image"):
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        st.write("")
        st.write("Detecting objects...")

        auto_select = st.radio("Model selection mode:", ('Automatic', 'Manual'))

        if auto_select == 'Manual':
            model_size = st.selectbox("Choose model size:", ('n', 's', 'm', 'b', 'l', 'x'))
            min_confidence = None
        else:
            min_confidence = st.slider("Select minimum confidence for automatic model selection:", 0.0, 1.0, 0.9)
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

                        detections_response = requests.post(
                            f'{backend_url}/get_detections',
                            files={'file': (filename, image_bytes, 'image/jpeg')},
                            data={
                                'auto_select': 'true' if auto_select == 'Automatic' else 'false',
                                'min_confidence': min_confidence if auto_select == 'Automatic' else None,
                                'model_size': model_size if auto_select == 'Manual' else None,
                                'target_language': target_language
                            }
                        )

                        if detections_response.status_code == 200:
                            detections_data = detections_response.json()
                            detections = detections_data.get('detections', [])
                            if detections:
                                visualize_decision_tree(detections, min_confidence, model_size)

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
                            st.error(
                                f"Error in detections. Status code: {detections_response.status_code}, Response content: {detections_response.text}")
                    except ValueError as e:
                        st.error(f"Error parsing detections response: {e}")
                else:
                    st.error(
                        f"Error in object detection. Status code: {response.status_code}, Response content: {response.text}")

    elif file_type.startswith("video"):
        st.video(uploaded_file)

        st.write("")
        st.write("Detecting objects in video...")

        auto_select = st.radio("Model selection mode:", ('Automatic', 'Manual'))

        if auto_select == 'Manual':
            model_size = st.selectbox("Choose model size:", ('n', 's', 'm', 'b', 'l', 'x'))
            min_confidence = None
        else:
            min_confidence = st.slider("Select minimum confidence for automatic model selection:", 0.0, 1.0, 0.9)
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
                    st.error(f"Error in video object detection. Status code: {response.status_code}, Response content: {response.text}")

else:
    st.info("Please upload an image or video.")

st.markdown("""
    ## API Documentation

    You can use this API to integrate object detection into your applications. The API provides endpoints to detect objects in images and videos.

    ### Decision Tree Logic

    The backend uses a decision tree to select the most appropriate YOLOv10 model based on the confidence intervals of detected objects. 

    **Model Selection Process:**

    1. **Set a minimum confidence threshold.**
    2. **Evaluate detections using each model size:**
       - If all detections meet the minimum confidence, select that model size.
       - If no models meet the confidence threshold, the largest model (`x`) is used.

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
""")
