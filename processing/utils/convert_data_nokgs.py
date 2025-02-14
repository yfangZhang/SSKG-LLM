import json,jsonlines
from utils.get_subkgs import get_relations
from utils.get_structure_kgs import get_structure_tokens
import random

# 读取原始 JSON 文件
with open('../truthfulqa.json', 'r', encoding='utf-8') as f:
    datas = json.load(f)
new_datas= []
# 为每条数据添加 subkg_sequence
for index,item in enumerate(datas):
    # sub_kgs = get_relations(item['question'])    
    # sub_kgs = random.sample(sub_kgs,min(5,len(sub_kgs)))
    # print('sub_kgs',sub_kgs)
    # path = []
    # if len(sub_kgs)!=0: 
        
    #     path = get_structure_tokens(sub_kgs)
    # print(index)
    # sub_kg_sequence = ' '.join(path)
    # item['subkg_sequence'] = sub_kg_sequence
    message = {"conversation_id": index+1,
               "category": "kg_qa",
               "conversation": [{"sub_kgs":"","human":"| Question is:"+item['question']+"| Choice is:"+item["choice"] ,"assistant": item['answer']}],
               "dataset": "truthfulqa"
               }
    # with jsonlines.open("wiki_zsl/wiki_multi_test_sample{}.jsonl".format(n), 'a') as w:
    #     w.write(message)
    with jsonlines.open("../truthfulqa_nokgs_0925.json", 'a') as w:
        w.write(message)
    new_datas.append(item)