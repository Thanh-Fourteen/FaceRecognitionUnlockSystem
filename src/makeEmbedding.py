import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

def save_file(dataset_path, embeddings_path, mtcnn, resnet):
    # Duyệt qua tất cả ảnh trong thư mục
    for filename in os.listdir(dataset_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(dataset_path, filename)
            img = Image.open(img_path)

            # Phát hiện khuôn mặt
            boxes, _ = mtcnn.detect(img)

            if boxes is not None:
                # Lấy khuôn mặt đầu tiên (giả sử chỉ có 1 khuôn mặt)
                face = boxes[0]
                x1, y1, x2, y2 = map(int, face)
                face_img = img.crop((x1, y1, x2, y2))

                # Nhúng khuôn mặt thành vector đặc trưng
                face_embedding = resnet(mtcnn(face_img).unsqueeze(0))

                # Lưu embedding vào file .pt
                torch.save(face_embedding, os.path.join(embeddings_path, f"{filename[:-4]}.pt"))
            else:
                print(f"Không tìm thấy khuôn mặt trong ảnh {filename}")

if __name__ == "__main__":
    # Khởi tạo MTCNN để phát hiện khuôn mặt và InceptionResnetV1 để nhúng
    #Nhận diện khuôn mặt
    mtcnn = MTCNN(image_size=160, margin=0)
    # Trích xuật đặc trưng
    # vggface2: Là tên của tập dữ liệu mà mô hình InceptionResnetV1 đã được huấn luyện trước
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    root = os.getcwd()
    dataset_path = os.path.join(root, "dataset", "image")
    embeddings_path = os.path.join(root, "dataset","embedding")

    save_file(dataset_path, embeddings_path, mtcnn, resnet)
    print("Done!")
    