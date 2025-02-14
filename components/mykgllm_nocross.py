import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple

from transformers import AutoModelForCausalLM,AutoModel,AutoTokenizer
from copy import deepcopy



class GrpahEncoder(nn.Module):
    def __init__(
        self,
        graphencoder_card,
        how,
    ) -> None:
        super(GrpahEncoder, self).__init__()
        self.how = how
        self.graph_model = AutoModel.from_pretrained(graphencoder_card, trust_remote_code=True)
        self.graph_tokenizer = AutoTokenizer.from_pretrained(graphencoder_card)
        # self.graph_model.shared.weight = deepcopy(nn.Parameter(self.graph_model.shared.weight.detach().clone().detach()))
        # self.graph_model.encoder.embed_tokens.weight = deepcopy(nn.Parameter(self.graph_model.encoder.embed_tokens.weight.detach().clone().detach()))
        self.outputs = []
        self.graph_model.requires_grad_(False)
    def forward(
        self,
        model_inputs:dict
    ):
        # print(id(self.graph_model.shared.weight), id(self.graph_model.encoder.embed_tokens.weight))
        outputs = self.graph_model(**model_inputs).last_hidden_state.detach().clone()
        return outputs

class KGAdapterCrossAttention(nn.Module):
    def __init__(self, kg_dim, llm_dim, num_heads=8):
        super(KGAdapterCrossAttention, self).__init__()
        self.kg_dim = kg_dim
        self.llm_dim = llm_dim
        self.kg_adapter = nn.Linear(self.kg_dim, self.llm_dim)
        # self.cross_attention = nn.MultiheadAttention(self.llm_dim, num_heads=num_heads)
    
    def forward(self, kg_embeds):
        # First apply the KG adapter to the KG embeddings
        batch_size, seq_len, _ = kg_embeds.shape
        kg_embeds = self.kg_adapter(kg_embeds.view(batch_size*seq_len, self.kg_dim)).reshape(batch_size, seq_len,self.llm_dim)
        # Return the attention output
        return kg_embeds
class MyKGLLM_nocross(nn.Module):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer:AutoTokenizer,
        graphencoder_card,
        kg_dim,
        llm_dim,
    ) -> None:
        super(MyKGLLM_nocross, self).__init__()
        self.kg_dim = kg_dim
        self.llm_dim = llm_dim
        self.llm = model
        self.tokenizer = tokenizer
        # self.kg_adapter = nn.Linear(self.kg_dim, self.llm_dim)
        self.graph_encoder = GrpahEncoder(graphencoder_card,how='global')
        self.graph_encoder.requires_grad_(False)

        # self.cross_attention = nn.MultiheadAttention(self.llm_dim, num_heads=8)
        self.kg_adapter_cross_attention = KGAdapterCrossAttention(kg_dim, llm_dim, num_heads=8)


    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        kg_ids: torch.LongTensor = None
    ):  
        # print(kg_ids)
        # kg_embeds = g_encoder(self.graph_model,kg_ids)
        kg_embeds = self.graph_encoder(kg_ids).clone()
        # batch_size, seq_len, _ = kg_embeds.shape
        # kg_embeds = self.kg_adapter(kg_embeds.view(batch_size*seq_len, self.kg_dim)).reshape(batch_size, seq_len,self.llm_dim)
        token_embeds = self.llm.model.base_model.embed_tokens(input_ids).clone()

        output = self.kg_adapter_cross_attention(kg_embeds)
        batch_size, seq_len, _ = output.shape
        # print("attn_output.shape",attn_output.shape)
        input_embeds = torch.cat((output, token_embeds), dim=1)

        # input_embeds = torch.cat((kg_embeds, token_embeds), dim=1)
        prefix_mask = torch.ones((batch_size, seq_len))
        prefix_labels = torch.full((batch_size, seq_len), fill_value=-100, dtype=torch.long)
        new_attention_mask = torch.cat((prefix_mask.cuda(), attention_mask), dim=-1)
        new_labels = torch.cat((prefix_labels.cuda(), labels), dim=-1)
        # print("input_embeds",input_embeds)
        # print("new_attention_mask",new_attention_mask)
        # print("new_labels",new_labels)
        return self.llm(
            input_ids=None,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


    def gradient_checkpointing_enable(self,**kwargs):
        if hasattr(self.llm, 'gradient_checkpointing_enable'):
            self.llm.gradient_checkpointing_enable(**kwargs)
    
    