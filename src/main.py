import os
import unclock 
from facenet_pytorch import MTCNN, InceptionResnetV1

if __name__ == "__main__":
    user = "Thanh"

    root = os.getcwd()
    path_file_embedding = os.path.join(root, "dataset", "embedding")
    mtcnn = MTCNN(image_size=160, margin=0)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    unlocked = unclock.unlock(path_file_embedding, mtcnn, resnet, user)

    if unlocked:
        print("Mở khóa điện thoại!")
    else:
        print("Không thể mở khóa!")
