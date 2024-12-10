import torch
import torch.nn.functional as F

def cosine_similarity_penalty(embeddings, labels, num_labels):
    penalty_loss = 0.2
    count = 0

    # 각 라벨 쌍에 대해 다른 라벨을 가진 샘플 간의 유사도 계산
    for label_a in range(num_labels):
        for label_b in range(label_a + 1, num_labels):
            label_a_mask = labels == label_a
            label_b_mask = labels == label_b
            label_a_embeddings = embeddings[label_a_mask]
            label_b_embeddings = embeddings[label_b_mask]

            # print(f"label_a: {label_a}, label_b: {label_b}")
            # print(f"label_a_mask: {label_a_mask}")
            # print(f"label_b_mask: {label_b_mask}")

            # print(f"label_a_embeddings size: {label_a_embeddings.size()}")
            # print(f"label_b_embeddings size: {label_b_embeddings.size()}")
            
            if label_a_embeddings.size(0) > 0 and label_b_embeddings.size(0) > 0:
                similarity_matrix = F.cosine_similarity(
                    label_a_embeddings.unsqueeze(1),  # (N_a, 1, D)
                    label_b_embeddings.unsqueeze(0),  # (1, N_b, D)
                    dim=-1
                )
                
                penalty = similarity_matrix.mean()
                penalty_loss += penalty
                count += label_a_embeddings.size(0) * label_b_embeddings.size(0)  # 비교 횟수

    if count > 0:
        penalty_loss = penalty_loss / count

    return penalty_loss
