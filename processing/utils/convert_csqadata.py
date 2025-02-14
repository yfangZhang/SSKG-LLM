import json,jsonlines
# from utils.get_structure_kgs import get_structure_tokens
from utils.get_structure_allkgs import get_structure_tokens
import random
is_split = True
has_kg = False
# 读取原始 JSON 文件
with open('../mysiqa_test_grpah.json', 'r') as file:
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
            
            _,path = get_structure_tokens(sub_kgs,text)
        # print(index)
        sub_kg_sequence = ';'.join(path)
        after = ''
        # item['subkg_sequence'] = sub_kg_sequence
        if len(sub_kg_sequence)>0:
            if is_split and has_kg:
                after = '_withkgs_split'
                message = {"conversation_id": index+1,
                        "category": "kg_qa",
                        "conversation": [{'kgs':sub_kg_sequence,"human":item['question'] ,"assistant": item['answer']}],
                        "dataset": "truthfulqa"
                    }
            elif not is_split and has_kg:
                after = '_withkgs'
                message = {"conversation_id": index+1,
                        "category": "kg_qa",
                        "conversation": [{"human":"| Knowledge_Graph is:"+sub_kg_sequence+"| Question is:"+item['question'] ,"assistant": item['answer']}],
                        "dataset": "truthfulqa"
                    }
            else:
                after = '_nokgs'
                message = {"conversation_id": index+1,
                        "category": "kg_qa",
                        "conversation": [{"human":"| Question is:"+item['question'] ,"assistant": item['answer']}],
                        "dataset": "truthfulqa"
                    }
            # with jsonlines.open("wiki_zsl/wiki_multi_test_sample{}.jsonl".format(n), 'a') as w:
            #     w.write(message)
            index+=1
            with jsonlines.open("../siqa_test{}.json".format(after), 'a') as w:
                w.write(message)
        