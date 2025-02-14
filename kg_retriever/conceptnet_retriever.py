import unicodedata
import numpy as np
import string
import logging
import nltk
import torch
from transformers import AutoTokenizer
from nltk.corpus import wordnet
import os
from nltk.corpus import wordnet as wn
import pickle
from nltk import word_tokenize
from nltk.corpus import stopwords
import json,jsonlines

class Conceptnet_retriever(object):
    def init(self, filepath):
        self.filepath = filepath
        self.concept2id = dict()
        self.stopwords = set(stopwords.words('english'))
        self.read_txt()

    def read_txt(self):
        f=open(self.filepath)
        concepts = f.readlines()  
        f.close()  # 关
        id = 0
        for concept in concepts:
            self.concept2id[concept.strip()] = id
            id += 1
    
    def lookup_concept_ids(self, text):
        #ents =  word_tokenize(text)   #分词
        ents =  text.split(" ")   #分词
        interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']  
        tokens = []
        words_ents_lists = []
        synonyms_list = []
        ent_list =[]
        for ent in ents:
            ent_name = []
            #if ent == "John" or ent == "probably" or en
            has_ent = False
            ent = ent.strip()
            if ent == "":
                continue
            if ent in interpunctuations:
                continue
            if ent.lower() in self.stopwords:
                continue 
            if ent in set(string.punctuation):
                #print('{} is punctuation, skipped!'.format(retrieve_token))
                continue
            ent_list.append(ent)
            words_ents_list =  [-1] * 5
            id_ent = self.concept2id.get(ent.lower(), -1)
            if id_ent == -1:
                continue
            #words_ents_list[0] = id_ent
            synonyms = []
            for syn in wordnet.synsets(ent):
                for lm in syn.lemmas():
                    synonyms.append(lm.name().lower())
            tmp = synonyms
            synonyms = list(set(tmp))
            #synonyms.sort(key = tmp.index) 
            synonyms.insert(0, ent.lower())
            i = 0
            j = 0
            synonyms_list.append(synonyms[0])
            while i < len(synonyms):
                id_ent = self.concept2id.get(synonyms[i].lower(), -1)
                if id_ent == -1:
                    i += 1
                    continue
                ent_name.append(synonyms[i].lower())
                i += 1
                words_ents_list[j] = id_ent
                has_ent = True
                j += 1
                if j >= 5:
                    break
            if has_ent:
                words_ents_lists.append(torch.IntTensor(words_ents_list))
                tokens.append(ent)
        # print('synonyms_list',synonyms_list)
        # print('ent_list',ent_list)
        return words_ents_lists, None, synonyms_list
    
def run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)

if __name__ == "__main__":
    # for syn in wordnet.synsets("table_tennis"):
    #     print('syn',syn)
    #     print(syn.lemmas())
    #     for lm in syn.lemmas():
    #         print(lm.name())
    retrievers = Conceptnet_retriever()
    retrievers.init("/root/zhangyf/KnowLA/data/kgs/conceptnet/concept.txt")
    with open('/root/zhangyf/KnowLA/data/data/SIQA/devc.json', 'r', encoding='utf-8') as file:
        datas = json.load(file)
    index = 0
    for data in datas:
        index += 1
        question_text = data["question"]
        choice_text = data["choices"]
        choice_text = ';'.join(choice_text)
        answer_text = data["answer"]
        #example = "street"
        words_ent_list, doc_conceptids2synset, qc_token = \
        retrievers.lookup_concept_ids(text=question_text+choice_text)
        words_ent_list, doc_conceptids2synset, ac_token = \
        retrievers.lookup_concept_ids(text=answer_text)
        message = {"index": index,"question":"| Question is:"+question_text+"| Choice is:"+choice_text,"answer":answer_text,"qc":qc_token,"ac":ac_token}
        with jsonlines.open("../SIQA_ground.json", 'a') as w:
            w.write(message)
        # print("qc_token",qc_token)
        # print("ac_token",ac_token)