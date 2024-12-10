
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocessing import label_mapping,tokenizing, load_and_augment_datasets, prepare_cc100dataset, prepare_data_with_reset_index
from transformers import DataCollatorWithPadding
import pandas as pd


# 데이터셋 로드
data_dir = '/content/drive/MyDrive/Colab Notebooks/project/dataset/'
pretrained_train = pd.read_csv(data_dir + 'train.csv', encoding='utf-8').reset_index(drop=True)
pretrained_val = pd.read_csv(data_dir + 'valid.csv', encoding='utf-8').reset_index(drop=True)
pretrained_test = pd.read_csv(data_dir+ 'test.csv', encoding='utf-8').reset_index(drop=True)

tl_train, tl_val, tl_test = prepare_cc100dataset(data_dir+'tl.csv',train_frac=0.8, val_frac=0.1, test_frac=0.1)
ko_train, ko_val, ko_test = prepare_cc100dataset(data_dir+'ko.csv',train_frac=0.8, val_frac=0.1, test_frac=0.1)
id_train, id_val, id_test = prepare_cc100dataset(data_dir+'id.csv',train_frac=0.8, val_frac=0.1, test_frac=0.1)
uz_train, uz_val, uz_test = prepare_cc100dataset(data_dir+'uz.csv',train_frac=0.8, val_frac=0.1, test_frac=0.1)

train_datasets = [pretrained_train, tl_train, ko_train, id_train, uz_train]
validation_datasets = [pretrained_val, tl_val, ko_val, id_val, uz_val]
test_datasets = [pretrained_test, tl_test, ko_test, id_test, uz_test]

augmented_train, augmented_val, test_data_df = load_and_augment_datasets(train_datasets,
                                                                      validation_datasets,
                                                                      test_datasets)  
                                                              
augmented_train = prepare_data_with_reset_index(augmented_train)
augmented_val = prepare_data_with_reset_index(augmented_val)
test_datas = prepare_data_with_reset_index(test_data_df)

augmented_train, train_labels = label_mapping(augmented_train, 'labels', 'labels')
augmented_val, val_labels = label_mapping(augmented_val, 'labels', 'labels')
test_dataset, test_labels = label_mapping(test_datas, 'labels', 'labels')

# 모델 로드
model_name = "microsoft/xtremedistil-l6-h384-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name,adding=True, truncation=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=13, ignore_mismatched_sizes=True)

tokenized_train = tokenizing(augmented_train, tokenizer, batch_size=100)
tokenized_val = tokenizing(augmented_val, tokenizer, batch_size=100)
tokenized_test = tokenizing(test_dataset, tokenizer,  batch_size=100)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)