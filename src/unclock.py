import cv2
import threading
from PIL import Image
import compareFace as cf


def process_frame(frame, mtcnn, resnet, embeddings_path, user, top_k, unlocked_event):
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face_img = frame[y1:y2, x1:x2]

            # Nhúng khuôn mặt
            detected_face = mtcnn(Image.fromarray(face_img))
            if detected_face is not None:
                face_embedding = resnet(detected_face.unsqueeze(0))

                # So sánh với các khuôn mặt đã lưu (song song)
                most_similar_faces = cf.compare_faces_parallel(face_embedding, embeddings_path, top_k)
                for name in most_similar_faces:
                    if name[:len(user)] != user:
                        return 
                unlocked_event.set()
                return


def unlock(embeddings_path, mtcnn, resnet, user, top_k = 3, limit_false = 15):
    cnt_false = 0
    frame_count = 0
    unlocked = False
    unlocked_event = threading.Event()
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 20 == 0: 
            process_thread = threading.Thread(target=process_frame, args=(frame, mtcnn, resnet, embeddings_path, user, top_k, unlocked_event))
            process_thread.start()
            
            # Hiển thị frame
            cv2.imshow('Face Unlock', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            process_thread.join()
            if process_thread.is_alive():
                continue  

            if unlocked_event.is_set():
                unlocked = True
            else:
                cnt_false += 1
                print(f"False {cnt_false}")

            # Kiểm tra điều kiện dừng
            if unlocked or cnt_false >= limit_false:
                break                    
        else:
            # Hiển thị frame
            cv2.imshow('Face Unlock', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Giải phóng camera
    video_capture.release()
    cv2.destroyAllWindows()

    return unlocked
