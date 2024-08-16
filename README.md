# AI Object Detection Framework and Educational Tool

![Project Logo](logo.webp)

![243 Languages Supported](https://img.shields.io/badge/Supports-243%20Languages-brightgreen)


## Description

This project is an AI-based object detection framework and educational tool that utilizes the YOLOv10 model for real-time object detection. It comprises a Python backend using Flask, a GUI application built with Tkinter, and a web frontend using Streamlit. The system enables users to upload images and videos for object detection and offers educational games for learning object names in 243 supported languages.

## Features

- **Object Detection**: Detect objects in images and videos using YOLOv10.
  - **Educational Game**: A game where users guess object names based on annotated images.
  - **Frontend Interface**: A web-based interface to upload files, select models, and view detection results.
  - **Language Support**: Translate object names to 243 languages with conversion capabilities.
  - **GUI Interface**: A graphical user interface for the non-website portion.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/faycalki/object-detection-educator.git
    ```

   2. Set up a Python virtual environment and install dependencies:
       ```bash
       python -m venv env
       source env/bin/activate  # On Windows use `env\Scripts\activate`
       pip install -r requirements.txt
       ```

   3. Run the backend server:
       ```bash
       python backend.py
       ```

   4. Start the frontend (optional):
       ```bash
       streamlit run frontend.py
       ```

## Usage

### GUI Application

1. Run the Tkinter GUI application:
    ```bash
    python educational_game.py
    ```

   2. Enter the server URL and select source and target languages.
   3. Upload an image to start the educational game.

### Web Frontend

1. Open the Streamlit web interface.
   2. Upload an image or video file.
   3. Choose the model size and language options.
   4. View detection results and download annotated files.

## Visual Workflow

To better understand the decision-making process within the frontend, refer to the following procedural diagrams:

### Frontend Procedural Tree

This diagram illustrates how decisions are processed from the moment the frontend is initiated:

![Frontend Procedural Tree](path/to/procedural_tree_diagram.png)

### Educational Game Procedures

This diagram provides a detailed view of how the game procedures work, outlining the flow from image upload to language selection:

![Educational Game Procedures](path/to/ortho_game_procedures.png)

## License

This project is licensed under the [GNU General Public License v3.0](https://opensource.org/licenses/GPL-3.0). See the [LICENSE](LICENSE) file for more details.

## TODO

- **Allow creations if the file size differs**: Implement support for different languages based on file size changes.
  - **Prevent duplicate requests**: Avoid hitting the detect endpoint twice for the same request. Currently, one request returns the annotated image and detections, while the other returns only detections.
  - **Frontend language selection**: Implement functionality to choose source and target languages.
  - **Non-English source language**: Complete implementation for source languages other than English.
  - **Video annotation language selection**: Allow choosing source and target languages for video annotation.

## Credits

- Developed by [Faycal Kilali](https://www.faycalkilali.com)
  - YOLOv10: Real-Time End-to-End Object Detection by Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, Guiguang Ding. ArXiv preprint arXiv:2405.14458, 2024

## Special Thanks

- Funded by TD Ignite Grants

## Contact

For any questions or feedback, please reach out to [Faycal Kilali](https://www.faycalkilali.com).
