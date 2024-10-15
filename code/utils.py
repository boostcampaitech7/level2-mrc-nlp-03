

# context에서 \\n을 공백으로 변환해주는 함수
def replace_newline_with_space(examples):
    examples['context'] = [context.replace('\\n', ' ') for context in examples['context']]
    return examples
