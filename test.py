
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from loading import data_dir
from preprocessing import label_mapping
from metrics import predict, confusion_matrix, cal_metrics, EvalPrediction, vis_umap
import datetime, time
import pandas as pd
import torch
import os
import wandb 

wandb.init(project="distilatedBert_test", entity="dmswl0707")
timestamp = datetime.datetime.now().strftime("%Y-%m%d-%H%M")

artifact_dir = './checkpoints/checkpoint-epoch-9.99/'
data_dir = './dataset/'
output_dir = './png/'

start_time = time.time()

# 테스트 데이터셋 로드
df = pd.read_excel(data_dir + 'input_data.xlsx')
df, language_labels = label_mapping(df, 'lang_code', 'label')
texts = df['text']
true_labels = df['label']

# 모델과 토크나이저 로드
model_name = "dmswl0707/Language_Recognition/run-remkdl15-history:v0"
artifact = wandb.use_artifact(model_name, type="wandb-history")
#artifact_dir = artifact.download()

config_path = os.path.join(artifact_dir, "config.json")
if not os.path.exists(config_path):
    print("Config file missing. Generating a new one...")
    config = AutoConfig.from_pretrained("microsoft/xtremedistil-l6-h384-uncased", num_labels=13)
    config.save_pretrained(artifact_dir)

model = AutoModelForSequenceClassification.from_pretrained(artifact_dir, num_labels=13,ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
#model.save_pretrained(artifact_dir)
#tokenizer.save_pretrained(artifact_dir)

#artifact.add_dir("./artitact_dir")
#wandb.log_artifact(artifact)

#tokenizer.save_pretrained(artifact_dir)
#model.save_pretrained(artifact_dir)

# 모델 성능 계산
model.eval()
predicted_labels = []
all_logits = [] 
for text in texts:
    predicted_label, logits = predict(text, model, tokenizer)  
    predicted_labels.extend(predicted_label)
    all_logits.append(logits)

logits_stack = torch.cat(all_logits, dim=0)

eval_pred = EvalPrediction(predictions=logits_stack.numpy(), label_ids=true_labels)
results = cal_metrics(eval_pred)
print("model result : ", results)

end_time = time.time()
print("test_time : ", end_time - start_time)

# 테스트 레이블 저장
df = pd.DataFrame(df) 
df['predicted'] = predicted_labels
output_path = os.path.join(data_dir, 'output_data.xlsx')
df.to_excel(output_path, index=False)
print(f"Output 데이터 저장: {output_path}")

# 혼동 행렬 계산
confusion_matrix(df, label_column='label', 
                 pred_column='predicted',  
                 labels=list(language_labels.values()), 
                 output_dir=data_dir, 
                 timestamp=timestamp)

# UMAP 시각화
vis_umap(logits_stack, true_labels, data_dir, timestamp)