import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import conceptnet_lite
from conceptnet_lite import Label, edges_for
from transformers import AutoTokenizer,AutoConfig
from nltk.corpus import wordnet
from get_keywords import ground_qa_pair
conceptnet_lite.connect('./conceptnet.db')

# 下载 nltk 的停用词
nltk.download('punkt')
nltk.download('stopwords')
def load_tokenizer(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' or config.model_type == 'internlm2' else True
    )
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
    return tokenizer
def extract_keywords(question):
    # stop_words = set(stopwords.words('english'))
    # tokenizer = load_tokenizer('/models/Qwen1.5-14B-Chat')
    # word_tokens = tokenizer.tokenize(question,add_special_tokens=False)
    # # print('word_tokens',word_tokens)
    # keywords = [word for word in word_tokens if word.isalnum() and word.lower() not in stop_words]
    keywords = ground_qa_pair(question,'')['qc']
    return keywords

def get_relations(question):
    relations = []
    keywords = extract_keywords(question)
    # print("keywords:",keywords)
    if keywords:
        # print(f"提取的关键词: {keywords}")
        for keyword in keywords:
            try:
                keyword_concepts = Label.get(text=keyword, language='en').concepts
                for e in edges_for(keyword_concepts, same_language=True):
                    rel_dic = dict()
                    rel_dic['head_entity'] = e.start.text
                    rel_dic['rel'] = e.relation.name
                    rel_dic['tail_entity'] = e.end.text
                    relations.append(rel_dic)
            except:
                try:
                    # print('keyword',keyword)
                    syns = wordnet.synsets(keyword)
                    for syn in syns:
                        syn_concepts = Label.get(text=syn, language='en').concepts
                        for e in edges_for(syn_concepts, same_language=True):
                            rel_dic = dict()
                            rel_dic['head_entity'] = e.start.text
                            rel_dic['rel'] = e.relation.name
                            rel_dic['tail_entity'] = e.end.text
                            relations.append(rel_dic)
                except:
                    print('Error, keyword',keyword)
            

    
    else:
        print("未找到关键词。")
    return relations

if __name__ == "__main__":
    question = "hello,i love you"
    sub_kgs = get_relations(question)
    print(sub_kgs)
