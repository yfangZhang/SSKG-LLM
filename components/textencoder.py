from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel
import torch
def load_qwen_encoder(modelcard):
    print('Load the model and tokenizer')
    model = AutoModelForCausalLM.from_pretrained(modelcard, trust_remote_code=True, revision='main')
    tokenizer = AutoTokenizer.from_pretrained(modelcard)
    return model.base_model,tokenizer

def text_encoder(model, tokenizer, text, max_length):
    # 文本预处理
    tokens = tokenizer.tokenize(text)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    if len(token_ids) < max_length:
        token_ids += [0] * (max_length - len(token_ids))
    else:
        token_ids = token_ids[:max_length]

    input_ids = torch.tensor([token_ids]).long()
    attention_mask = torch.ones_like(input_ids)

    # 文本向量化
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    return outputs,token_ids

def text_qwen_encoder(model, tokenizer, text, max_length):
    device = torch.device("cuda")
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    print('input_ids.device',input_ids.device)
    # print('q_model.device',model.device)
    outputs = model(input_ids)
    return outputs,input_ids

if __name__ == "__main__":
    modelcard = '/models/Qwen2-0.5B-Instruct'
    text = 'summarize: The black poodle chased the cat.' 
    model, tokenizer = load_qwen_encoder(modelcard)
    print(model.config)
    outputs,input_ids= text_qwen_encoder(model, tokenizer,text= text,max_length=256)
    print('input_ids',input_ids)
    print('prepare model inputs')
    # get token embeddings
    print('Sequence of tokens (batch_size, max_seq_len, embedding_dim):', outputs.shape)  # embeddings of all graph and text tokens. Nodes in the graph (e.g., dog) appear only once in the sequence.
  