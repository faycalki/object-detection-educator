# AI Object Detection Framework and Educational Tool

![Project Logo](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMDAgMjAwIiB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCI+CiAgPHBhdGggZD0iTTYwIDYwIEwxNDAgNjAgMTAwIDE0MCIgZmlsbD0iIzAwN0JGRiIgLXN0cm9rZT0iIzAwMCIgc3Ryb2tlLXdpZHRoPSIyIi8+CiAgPGNpcmNsZSBjeD0iMTAwIiBjeT0iMTAwIiByPSIxMCIgZmlsbD0iI2ZmZiIvPgogIDx0ZXh0IHg9IjUwJSIgeT0iMTgwIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9IiMwMDAiIHRleHQtYW5jaG9yPSJtaWRkbGUiPkFJIE9iamVjdCBkZXRlY3Rpb248L3RleHQ+CiAgPHRleHQgeD0iNTAlIiB5PSIxOTUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxMiIgZmlsbD0iIzAwMCIgdGV4dC1hbmNob3I9Im1pZGRsZSI+RnJhbWV3b3JrICYgVG9vbDwvdGV4dD4KPC9zdmc+Cg==)

## Description

This project is an AI-based object detection framework and educational tool that utilizes the YOLOv10 model for real-time object detection. It comprises a Python backend, a GUI application built with Tkinter, and a web frontend using Streamlit. The system enables users to upload images and videos for object detection and offers educational games for learning object names in 243 supported languages.

## Features

- **Object Detection**: Detect objects in images and videos using YOLOv10.
- **Educational Game**: A game where users guess object names based on annotated images.
- **Frontend Interface**: A web-based interface to upload files, select models, and view detection results.
- **Language Support**: Translate object names to 243 languages with conversion capabilities.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/yourproject.git
    cd yourproject
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

4. Start the frontend:
    ```bash
    streamlit run frontend.py
    ```

## Usage

### GUI Application

1. Run the Tkinter GUI application:
    ```bash
    python gui_app.py
    ```

2. Enter the server URL and select source and target languages.
3. Upload an image to start the educational game.

### Web Frontend

1. Open the Streamlit web interface.
2. Upload an image or video file.
3. Choose the model size and language options.
4. View detection results and download annotated files.

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
