from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import compareFace as cf 

app = Flask(__name__)
CORS(app)  # Kích hoạt CORS cho tất cả các endpoint

# Khởi tạo model
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
root = os.getcwd()
embeddings_path = os.path.join(root, "dataset", "embedding")

@app.route('/') # Endpoint cho trang chủ
def home():
    return 'Face Recognition API 2'

@app.route('/unlock', methods=['POST'])
def unlock_api():
    try:
        # Kiểm tra xem có file ảnh được upload không
        if 'image' not in request.files:
            return jsonify({"status": "no_file_uploaded"})

        file = request.files['image']
        img = Image.open(file.stream)
        
        # Phát hiện khuôn mặt và nhúng
        boxes, _ = mtcnn.detect(img)
        if boxes is not None:
            face = boxes[0]
            x1, y1, x2, y2 = map(int, face)
            face_img = img.crop((x1, y1, x2, y2))
            face_embedding = resnet(mtcnn(face_img).unsqueeze(0))
            
            # So sánh với embeddings đã lưu
            top_k = 3  # Chọn số lượng khuôn mặt được so sánh
            most_similar_faces = cf.compare_faces_parallel(face_embedding, embeddings_path, top_k)
            
            # Kiểm tra xem có khớp với khuôn mặt nào trong embeddings không
            for name in most_similar_faces:
                if name[:5] == "Thanh": # Thay thế "your_user_name" bằng tên người dùng
                    return jsonify({"status": "unlocked"}) 
            return jsonify({"status": "locked"}) 
        else:
            return jsonify({"status": "no_face_detected"}) 
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # app.run(debug=True) 
    app.run(host='0.0.0.0', port='6868')
