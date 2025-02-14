import json,jsonlines
def read_line(filename, line_number):
    with open(filename, 'r') as file:
        for current_line, content in enumerate(file, start=1):
            if current_line == line_number:
                return content
    return None  # 如果行号超出文件范围

my_dict = {chr(65 + i): i for i in range(4)}
def read_and_process_json_line_by_line(input_file_path,source_path, output_file_path):
    results = []  # 用于存储所有结果的列表
    batch_size = 4
    qa_data = []
    total_index = 0

    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # 确保不处理空行
                entry = json.loads(line)
                total_index += 1
                q_ids = set(entry['qc'])
                a_ids = set(entry['ac'])
                q_ids -= a_ids
                qa_data.append((q_ids, a_ids))

                if (total_index % batch_size == 0):
                    index = total_index/batch_size
                    choice_str = ''
                    for i in range(4):
                        choice_str+=json.loads(read_line(source_path, index))["question"]["choices"][i]['text']+';'
                    an_label = my_dict[json.loads(read_line(source_path, index))["answerKey"]]
                    read_line(source_path, index)
                    result = {
                        'index': int(index),
                        'question':'| Question is:'+json.loads(read_line(source_path, index))["question"]['stem']+'| Choice is:'+choice_str,
                        'answer':json.loads(read_line(source_path, index))["question"]["choices"][an_label]['text'],
                        'qc': list(set.union(*[q for q, a in qa_data])),
                        'ac': list(set.union(*[a for q, a in qa_data]))
                    }
                    with jsonlines.open(output_file_path, 'a') as w:
                        w.write(result)
                    results.append(result)  # 将结果添加到结果列表
                    qa_data = []  # 清空列表以便下一批处理



# 调用函数，传入输入文件路径和输出文件路径
read_and_process_json_line_by_line('/data/obqa/grounded/dev.grounded.jsonl', '/data/obqa/statement/dev.statement.jsonl','/data/obqa/merge_dev.grounded.jsonl')