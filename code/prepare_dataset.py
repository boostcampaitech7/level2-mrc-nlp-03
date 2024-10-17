from datasets import DatasetDict, load_from_disk, load_dataset, concatenate_datasets
from utils import replace_newline_with_space, parse_answer_column


def prepare_dataset(data_type, train_dataset_name, newline_to_space):
    """dataset load 및 return하는 function

    Args:
        data_type: args에서 받아오는 값
            original: 기존 데이터셋
            korquad: korquad v1 데이터셋
                train: korquad_train
                valid: korquad_valid
            korquad_hard: korquad의 valid까지 train으로 사용
                train: korquad_train + korquad_valid
                valid: original_valid
                
        train_dataset_name: original dataset 경로

        newline_to_space: True일 경우, \\n을 공백으로 바꿔줌. (original dataset에서만)

    Returns:
        DatasetDict: dataset
    """
    if data_type == 'original':
        # train_dataset_name에 csv가 포함되어 있으면 csv 파일을 load
        if 'csv' in train_dataset_name:
            train_dataset = load_dataset('csv', data_files={"train": train_dataset_path})['train']
            validation_dataset = load_dataset('csv', data_files={"validation": validation_dataset_path})['validation']
            
            train_dataset = train_dataset.map(parse_answer_column, batched=True)
            validation_dataset = validation_dataset.map(parse_answer_column, batched=True)
            return DatasetDict({"train": train_dataset, "validation": validation_dataset})

        # train_dataset_name에 csv가 포함되어 있지 않으면 load_from_disk로 load
        # newline_to_space가 True일 경우, \\n을 공백으로 바꿔줌
        if newline_to_space:
            dataset = load_from_disk(train_dataset_name)
            processed_data_no_newlines_tr = dataset['train'].map(replace_newline_with_space, batched=True)
            processed_data_no_newlines_val = dataset['validation'].map(replace_newline_with_space, batched=True)
            return DatasetDict({'train': processed_data_no_newlines_tr, 'validation': processed_data_no_newlines_val})
        else: # newline_to_space가 False일 경우
            return load_from_disk(train_dataset_name)
    
    elif data_type == 'korquad':
        return load_dataset("squad_kor_v1")
    
    elif data_type == 'korquad_hard':
        # korquad_valid까지 train으로 사용
        original_datasets = load_from_disk(train_dataset_name)
        original_datasets['train'] = original_datasets['train'].remove_columns(['document_id', '__index_level_0__'])
        
        korquad_datasets = load_dataset("squad_kor_v1", features=original_datasets["train"].features)
        
        train_dataset = concatenate_datasets([korquad_datasets['train'], korquad_datasets['validation']])
        validation_dataset = original_datasets['validation'].remove_columns(['document_id', '__index_level_0__'])
        
        return DatasetDict({'train': train_dataset, 'validation': validation_dataset})
    
    else:
        raise ValueError("Invalid data_type")