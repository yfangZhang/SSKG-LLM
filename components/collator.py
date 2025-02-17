from typing import Any, Dict, List
import torch
import torch.nn as nn
from loguru import logger
from .graphencoder import graph_to_batch
class SFTDataCollator(object):
    def __init__(self, tokenizer, graph_tokenizer,graph_model,max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.graph_tokenizer = graph_tokenizer
        self.graph_model = graph_model

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 找出batch中的最大长度
        lengths = [len(x['input_ids']) for x in batch if x['input_ids'] is not None]
        # kg_lengths = [len(x['kg_ids']) for x in batch if x['kg_ids'] is not None]
        # 取出batch中的最大长度，如果超过max_seq_length，则取max_seq_length
        batch_max_len = min(max(lengths), self.max_seq_length)
        # batch_max_kg_len = min(max(kg_lengths), self.max_seq_length)
        # batch_max_len = self.max_seq_length

        input_ids_batch, attention_mask_batch, target_mask_batch, kg_ids_batch = [], [], [], []
        for x in batch:
            input_ids = x['input_ids']
            kg_ids = x['kg_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']
            if input_ids is None:
                logger.info('some input_ids is None')
                continue
            padding_len = batch_max_len - len(input_ids)
            # kg_padding_len = batch_max_kg_len - len(kg_ids)

            input_ids = input_ids + [self.pad_token_id] * padding_len
            # kg_ids = kg_ids + [self.pad_token_id] * kg_padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len 

            input_ids = input_ids[:self.max_seq_length]
            # kg_ids = kg_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            target_mask = target_mask[:self.max_seq_length]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)
            kg_ids_batch.append(kg_ids)
            

        # 将list转换为tensor，得到最终的的模型输入
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)
        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        # kg_ids_batch = torch.tensor(kg_ids_batch)
        # print("kg_ids_batch.shape",kg_ids_batch.shape)
        kg_ids_batch = graph_to_batch(self.graph_tokenizer,self.graph_model,kg_ids_batch)
        # kg_ids_batch = torch.tensor(kg_ids_batch, dtype=torch.long)
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels,
            'kg_ids':kg_ids_batch
        }
        return inputs


class PretrainCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = [x['input_ids'] for x in batch if x['input_ids'] is not None]
        # 找出batch中的最大长度
        lengths = [len(x) for x in batch]
        # 取出batch中的最大长度，如果超过max_seq_length，则取max_seq_length
        batch_max_len = min(max(lengths), self.max_seq_length)
        # batch_max_len = self.max_seq_length

        input_ids_batch, attention_mask_batch, labels_batch = [], [], []
        for x in batch:
            input_ids = x
            attention_mask = [1] * len(input_ids)

            padding_len = batch_max_len - len(input_ids)
            # padding
            labels = input_ids + [-100] * padding_len
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            # truncate
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)
            attention_mask_batch.append(attention_mask)

        # 将list转换为tensor，得到最终的的模型输入
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels_batch
        }
        return inputs

