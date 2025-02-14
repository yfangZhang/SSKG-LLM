import json,jsonlines
from utils.get_structure_allkgs import get_structure_tokens
import random

# 读取原始 JSON 文件
with open('../myobqa_train_grpah.json', 'r') as file:
    # 逐行读取
    index = 0
    for line in file:
        # 解析JSON
        item = json.loads(line)
        sub_kgs = item["subkgs"]
        text = item['question'].replace('| Question is:','Question: ').replace('| Choice is:',' Choice: ').replace(';',' ')
        # sub_kgs = get_relations(item["subkgs"])    
        # sub_kgs = random.sample(sub_kgs,min(5,len(sub_kgs)))
        # print('sub_kgs',sub_kgs)
        path = []
        if len(sub_kgs)!=0: 
            
            path,_ = get_structure_tokens(sub_kgs,text)
        path = path[:-1]
        # print(index)
        sub_kg_sequence = ' '.join(path)
        # if index == 0:
        #     print(sub_kg_sequence)
        # item['subkg_sequence'] = sub_kg_sequence
        message = {"conversation_id": index+1,
                "category": "kg_qa",
                "conversation": [{"human":sub_kg_sequence.replace(' <next_token>','') ,"assistant": item['answer']}],
                "dataset": "csqa"
                }
        # with jsonlines.open("wiki_zsl/wiki_multi_test_sample{}.jsonl".format(n), 'a') as w:
        #     w.write(message)
        index+=1
        with jsonlines.open("../obqa_train_allkgs4_nonext.json", 'a') as w:
            w.write(message)
        