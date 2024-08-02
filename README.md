## Face Recognition Unlock System

This project implements a simple face recognition unlock system using Python and OpenCV.

### Features

* **Face Detection:** Detects faces in real-time using MTCNN.
* **Face Recognition:** Extracts facial features using InceptionResnetV1 (pre-trained on VGGFace2) and compares them for recognition.
* **User Authentication:** Allows unlocking only for authorized users.
* **Parallel Processing:** Utilizes threading to improve performance by processing frames concurrently.

### Requirements

* Python 3.9.19
* OpenCV (`pip install opencv-python`)
* facenet-pytorch (`pip install facenet-pytorch`)

### Installation

1. Clone the repository:
```
git clone https://github.com/your-username/face-recognition-unlock.git
```

2. Install the required libraries:
```
pip install -r requirements.txt
```

### Usage

1. **Dataset Preparation:**
    * Create a folder named `dataset/image` in the project directory.
    * Place images of authorized users in this folder. Each user should have a separate subfolder with their name.
    * Example folder structure:
        ```
        dataset/
        ├── image/
        │   ├── user1/
        │   │   ├── image1.jpg
        │   │   └── image2.png
        │   └── user2/
        │       ├── image1.jpeg
        │       └── image2.jpg
        └── embedding/

        ```

2. **Embedding Generation:**
    * Run `makeEmbedding.py` to generate face embeddings for the images in the dataset.
    * This script will create a folder named `dataset/embedding` and store the embeddings as `.pt` files.

3. **Running the System:**
    * Modify `main.py` to specify the authorized user's name.
    * Run `main.py` to start the face recognition unlock system.
    * The system will use your webcam to capture video.
    * If the recognized face matches the authorized user, the system will unlock.

### Configuration

* `top_k` (in `unclock.py`): The number of most similar faces to consider for recognition.
* `limit_false` (in `unclock.py`): The maximum number of consecutive false recognitions allowed before the system stops.

### Notes

* The system's accuracy and performance may vary depending on factors such as lighting conditions and image quality.
* The provided code is a basic implementation and can be further improved and customized.

### Disclaimer

This project is for educational purposes only and should not be used for security-critical applications without proper testing and evaluation.
