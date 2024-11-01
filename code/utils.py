from datasets import load_from_disk

# context에서 \\n을 공백으로 변환해주는 함수
def replace_newline_with_space(examples):
    # answer_start 조정
    for i, answer in enumerate(examples['answers']):
        answer_start = answer['answer_start'][0]
        new_line_count = (examples['context'][i][:answer_start]).count('\\n')
        new_answer_start = answer_start - new_line_count # \\n이 ' '으로 대체되므로 1개당 -1. (\\n은 2글자입니다..)
        answer['answer_start'][0] = new_answer_start
    
    # context 처리
    examples['context'] = [context.replace('\\n', ' ') for context in examples['context']]
    return examples

# context에서 \\n 뒤를 띄어쓰기하는 함수
def add_space_after_newline(examples):
    # answer_start 조정
    for i, answer in enumerate(examples['answers']):
        answer_start = answer['answer_start'][0]
        new_line_count = (examples['context'][i][:answer_start]).count('\\n')
        new_answer_start = answer_start + new_line_count # \\n이 ' '으로 대체되므로 1개당 -1. (\\n은 2글자입니다..)
        answer['answer_start'][0] = new_answer_start
    
    # context 처리
    examples['context'] = [context.replace('\\n', '\\n ') for context in examples['context']]
    return examples

# 'answers' 열이 문자열인 경우, ast.literal_eval을 사용하여 변환
def parse_answer_column(examples):
    if isinstance(examples['answers'][0], str):
        examples['answers'] = [ast.literal_eval(answer) for answer in examples['answers']]
    return examples


if __name__ == "__main__":
    # 함수 테스트입니다. 
    dataset = load_from_disk('../data/train_dataset')
    processed_data_no_newlines_tr = dataset['train'].map(add_space_after_newline, batched=True)

    print('*** original dataset ***')
    for answer in dataset['train']['answers'][:5]:
        print(answer)

    print('*** no_newline dataset ***')
    for answer in processed_data_no_newlines_tr['answers'][:5]:
        print(answer)

    print('*** answer_start correctness test ***')
    for i, answer in enumerate(processed_data_no_newlines_tr['answers'][:5]):
        ans_start = answer['answer_start'][0]
        length = len(answer['text'][0])
        
        print(processed_data_no_newlines_tr['context'][i][ans_start:ans_start+length])
        

