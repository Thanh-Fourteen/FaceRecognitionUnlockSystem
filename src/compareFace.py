import os
import math
import torch
import numpy as np

def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        dist = 0
        raise 'Undefined distance metric %d' % distance_metric

    print(dist)
    return dist

def compare_faces_parallel(embedding1, embeddings_path, top_k, threshold=0.5):
    """
    So sánh một embedding khuôn mặt với nhiều embedding khác song song.

    Args:
        embedding1 (torch.Tensor): Embedding của khuôn mặt cần so sánh.
        embeddings_path (str): Đường dẫn đến thư mục chứa các file embedding.
        top_k (int): Số lượng file có cosine similarity cao nhất được trả về.
        threshold (float): Giá trị ngưỡng của cosine similarity để lọc kết quả.

    Returns:
        list: Danh sách top_k tên file có cosine similarity cao nhất sau khi lọc theo ngưỡng.
    """
    file_names = []
    similarities = []

    for filename in os.listdir(embeddings_path):
        if filename.endswith('.pt'):
            embedding2 = torch.load(os.path.join(embeddings_path, filename))
            cosine_similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
            similarity_value = cosine_similarity.item()

            if similarity_value >= threshold:
                similarities.append(similarity_value)
                file_names.append(filename[:-3])

    # Lấy top_k chỉ số có cosine similarity cao nhất
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Trả về top_k tên file tương ứng
    return [file_names[i] for i in top_k_indices]
