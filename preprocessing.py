from datasets import  Dataset, concatenate_datasets
import pandas as pd
import random
import re

def label_mapping(df, lang_column, label_column, num_labels=13):
    """lan_code를 index로 변환"""
    
    if isinstance(df, list):
        df = pd.DataFrame(df, columns=[lang_column])
        
    if isinstance(df, pd.DataFrame):
        df = Dataset.from_pandas(df)
    elif not isinstance(df, Dataset):
        raise TypeError("Expected input to be a Dataset, DataFrame, or list.")

    unique_langs = sorted(set(df[lang_column]))  
    lang_to_label = {lang: idx for idx, lang in enumerate(unique_langs) if idx < num_labels}

    df = df.map(lambda x: {label_column: lang_to_label.get(x[lang_column], None)})
    df = df.filter(lambda x: x[label_column] is not None) 
    return df, lang_to_label

def tokenizing(data, tokenizer, batch_size=32, max_length=512):
    """텍스트 토크나이징"""
    if isinstance(data, Dataset):
        texts = data["text"]
        labels = data["labels"]
    elif isinstance(data, pd.DataFrame) and "text" in data.columns:
        texts = data["text"].tolist()
        labels = data["labels"].tolist() if "labels" in data.columns else None
    elif isinstance(data, list):
        texts = data
        labels = None
    else:
        raise ValueError("Input data should be a DataFrame with 'text' and optionally 'labels' columns, or a Dataset")
    
    all_tokenized_data = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_labels = labels[i:i + batch_size] if labels else [None] * len(batch_texts)

        if all(isinstance(text, str) for text in batch_texts):
            tokenized_batch = tokenizer(
                batch_texts, 
                padding="max_length", 
                truncation=True, 
                max_length=max_length
            )
            all_tokenized_data["input_ids"].extend(tokenized_batch["input_ids"])
            all_tokenized_data["attention_mask"].extend(tokenized_batch["attention_mask"])
            all_tokenized_data["labels"].extend(batch_labels)

    return Dataset.from_dict(all_tokenized_data)

def remove_stopword(text):
    """특수문자 및 숫자 제거"""
    if not isinstance(text, str):
        return ""
    
    clean_text = re.sub(r'[^가-힣a-zA-Z\s]', '', text)
    return clean_text

def add_typo_mistake(text, error_rate=0.1):
    """철자 실수 노이즈"""
    text = list(text)
    for i in range(len(text)):
        if random.random() < error_rate:
            swap_index = random.randint(0, len(text) - 1)
            text[i], text[swap_index] = text[swap_index], text[i]  # 문자 위치 변경
    return ''.join(text)

def add_delete(text, deletion_rate=0.1):
    """철자 삭제 노이즈"""
    text = list(text)
    i = 0
    while i < len(text):
        if random.random() < deletion_rate:
            del text[i]  # 문자를 완전히 삭제
        else:
            i += 1  # 삭제되지 않으면 다음 문자로 이동
    return ''.join(text)

def reverse_sentence(text):
    """문장 역순 노이즈"""
    words = text.split()
    return ' '.join(reversed(words))

def add_noise(text):
    """노이즈 적용함수"""
    text = add_typo_mistake(text, error_rate=0.2)   
    text = add_delete(text,deletion_rate=0.1)
    text = reverse_sentence(text) if random.random() < 0.1 else text
    return text

def augment_text(df):
    """특수문자 및 숫자 제거 + 노이즈 추가"""
    df['text'] = df['text'].apply(remove_stopword)
    df['text'] = df['text'].apply(add_noise)
    return df

def convert_to_dataframe(dataset):
    """데이터 프레임 변환"""
    return pd.DataFrame(dataset)

def prepare_data_with_reset_index(df):
    """인덱스 정리"""
    df = df.reset_index(drop=True)
    return Dataset.from_pandas(df)

def load_and_augment_datasets(train_datasets, val_datasets, test_datasets):
    """데이터셋 병합 + augment 추가"""
    train_datasets = [Dataset.from_pandas(df.reset_index(drop=True)) for df in train_datasets]
    val_datasets = [Dataset.from_pandas(df.reset_index(drop=True)) for df in val_datasets]
    test_datasets = [Dataset.from_pandas(df.reset_index(drop=True)) for df in test_datasets]
    
    train_dataset = concatenate_datasets(train_datasets)
    validation_dataset = concatenate_datasets(val_datasets)
    test_dataset = concatenate_datasets(test_datasets)
    
    train_dataset_df = convert_to_dataframe(train_dataset)
    val_dataset_df = convert_to_dataframe(validation_dataset)
    test_dataset_df = convert_to_dataframe(test_dataset)

    # 레이블 구성 검증
    label_counts = train_dataset_df['labels'].value_counts()
    # print(label_counts)

    label_ratios = label_counts / len(train_dataset_df)
    # print("클래스 비율:\n", label_ratios)
    
    augmented_train = augment_text(train_dataset_df.reset_index(drop=True))
    augmented_val = augment_text(val_dataset_df.reset_index(drop=True))
    
    return augmented_train, augmented_val, test_dataset_df

def prepare_cc100dataset(file_path, train_frac=0.8, val_frac=0.1, test_frac=0.1):
    """cc100 데이터셋 변환"""
    dataset = pd.read_csv(file_path)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_end = int(train_frac * len(dataset))
    val_end = train_end + int(val_frac * len(dataset))
    
    train_dataset = dataset[:train_end]
    val_dataset = dataset[train_end:val_end]
    test_dataset = dataset[val_end:]
    
    return train_dataset, val_dataset, test_dataset
