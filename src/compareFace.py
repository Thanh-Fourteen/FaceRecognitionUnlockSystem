import os
import torch
import numpy as np

def compare_faces_parallel(embedding1, embeddings_path, top_k):
    """
    So sánh một embedding khuôn mặt với nhiều embedding khác song song.

    Args:
        embedding1 (torch.Tensor): Embedding của khuôn mặt cần so sánh.
        embeddings_path (str): Đường dẫn đến thư mục chứa các file embedding.
        top_k (int): Số lượng file có cosine similarity cao nhất được trả về.

    Returns:
        list: Danh sách top_k tên file có cosine similarity cao nhất.
    """
    file_names = []
    similarities = []

    for filename in os.listdir(embeddings_path):
        if filename.endswith('.pt'):
            embedding2 = torch.load(os.path.join(embeddings_path, filename))
            cosine_similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
            file_names.append(filename[:-3])
            similarities.append(cosine_similarity.item())

    # Lấy top_k chỉ số có cosine similarity cao nhất
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Trả về top_k tên file tương ứng
    return [file_names[i] for i in top_k_indices]
