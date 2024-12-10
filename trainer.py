from transformers import Trainer
from loss import cosine_similarity_penalty
import torch.nn.functional as F

penalty_weight = 0.01  

class CustomTrainer(Trainer):
    """커스텀 트레이너 구성"""
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        inputs['input_ids'] = inputs['input_ids'].view(-1, inputs['input_ids'].size(-1))
        inputs['attention_mask'] = inputs['attention_mask'].view(-1, inputs['attention_mask'].size(-1))
        
        # 데이터 타입 검증
        if 'token_type_ids' in inputs:
            inputs['token_type_ids'] = inputs['token_type_ids'].view(-1, inputs['token_type_ids'].size(-1))

        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Labels are missing from the inputs. Ensure that your dataset includes a 'labels' column.")

        labels = labels.to(model.device).long()
        if labels.max() >= model.config.num_labels:
            raise ValueError(f"라벨 값 {labels.max()}가 'num_labels' {model.config.num_labels} 범위를 초과합니다.")
        
        outputs = model(**inputs)
        logits = outputs.logits

        loss = F.cross_entropy(logits, labels)

        # 페널티 손실 계산
        embeddings = model.get_input_embeddings()(inputs['input_ids'])
        penalty_loss = cosine_similarity_penalty(embeddings, labels, num_labels=model.config.num_labels)

        total_loss = loss + penalty_weight * penalty_loss
        return (total_loss, outputs) if return_outputs else total_loss