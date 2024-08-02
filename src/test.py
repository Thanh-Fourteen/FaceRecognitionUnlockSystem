import cv2
import torch
from PIL import Image
import compareFace as cf

def unlock(path_file_embedding, mtcnn, resnet):
    video_capture = cv2.VideoCapture(0)
    # your_face_embedding = torch.load('embeddings/your_face.pt')
    your_face_embedding = torch.load(path_file_embedding)
    unlocked = False

    while True:
        _, frame = video_capture.read()
        boxes, _ = mtcnn.detect(frame)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face_img = frame[y1:y2, x1:x2]

                # Nhúng khuôn mặt
                detected_face = mtcnn(Image.fromarray(face_img))
                if detected_face is not None:
                    face_embedding = resnet(detected_face.unsqueeze(0))
                else:
                    break

                # So sánh với khuôn mặt đã lưu
                if (cf.compare_faces(your_face_embedding, face_embedding) > 0.8):
                    unlocked = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, 'Unlocked', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    break
                else:
                    unlocked = False

        cv2.imshow('Face Unlock', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng camera
    video_capture.release()
    cv2.destroyAllWindows()

    return unlocked
    
def exc_thread(frame, embeddings_path, mtcnn, resnet, user, top_k, cnt_false, limit_false):
    global unlocked
    # Tạo luồng để xử lý frame                                   
    process_thread = threading.Thread(target=process_frame, args=(frame, mtcnn, resnet, embeddings_path, user, top_k))
    process_thread.start()
    
    # Hiển thị frame
    cv2.imshow('Face Unlock', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return 0, cnt_false, unlocked
    
    # Đợi luồng xử lý frame kết thúc
    process_thread.join()
    if process_thread.is_alive():
        return 1, cnt_false, unlocked  
    unlocked = True if unlocked else False
    if not unlocked:
        cnt_false += 1
        print(f"False {cnt_false}") 
    if((unlocked == True) or (cnt_false >= limit_false)):
        return 0, cnt_false, unlocked 
    return 1, cnt_false, unlocked

def exc_thread(frame, embeddings_path, mtcnn, resnet, user, top_k, cnt_false, limit_false):
    global unlocked
    # Tạo luồng để xử lý frame                                   
    process_thread = threading.Thread(target=process_frame, args=(frame, mtcnn, resnet, embeddings_path, user, top_k))
    process_thread.start()
    
    # Hiển thị frame
    cv2.imshow('Face Unlock', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return 0, cnt_false, unlocked
    
    # Đợi luồng xử lý frame kết thúc
    process_thread.join()

    # Loại bỏ kiểm tra is_alive()
    # unlocked = True if unlocked else False 

    if not unlocked:
        cnt_false += 1
        print(f"False {cnt_false}") 
    if((unlocked == True) or (cnt_false >= limit_false)):
        return 0, cnt_false, unlocked 
    return 1, cnt_false, unlocked

# status, cnt_false, unlocked = exc_thread(frame, embeddings_path, mtcnn, resnet, user, top_k, cnt_false, limit_false) 
            # print(status, cnt_false, unlocked)
            # if status: continue
            # else: break