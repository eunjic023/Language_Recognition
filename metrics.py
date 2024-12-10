
from transformers import EvalPrediction
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import numpy as np
import umap
import os

def confusion_matrix(df, label_column, pred_column, labels, output_dir, timestamp):
    """테스트파일 추론결과 CM으로 변환"""
    y_true = df[label_column].tolist()
    y_pred = df[pred_column].tolist()
    cm = sk_confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    cm_filename = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
    plt.savefig(cm_filename)
    plt.close()

def vis_umap(logits_stack, labels, output_dir, timestamp):
    """
    UMAP을 사용해 고차원 데이터를 저차원으로 축소하고 시각화
    """
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embeddings = reducer.fit_transform(logits_stack.numpy())

    # UMAP 결과 시각화
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7,
        edgecolor="k"
    )
    plt.colorbar(scatter, label="Labels")
    plt.title("UMAP plot of Logits")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True)

    output_path = os.path.join(output_dir, f'umap_visualization_{timestamp}.png')
    plt.savefig(output_path)
    plt.show()


def cal_metrics(eval_pred: EvalPrediction):
    """acc, precision, recall, f1 score for train"""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predic = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predic)
    precision = precision_score(labels, predic, average="weighted")
    recall = recall_score(labels, predic, average="weighted")
    f1 = f1_score(labels, predic, average="weighted")
    results = {"accuracy": acc,
               "precision": precision,
               "recall": recall,
               "f1": f1
               }
    return results

def predict(text, model, tokenizer):
    """텍스트를 입력받아 예측 레이블 반환"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    predicted_label = torch.argmax(logits, dim=1).tolist()

    return predicted_label, logits
    

