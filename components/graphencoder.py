from transformers import AutoTokenizer, AutoModel
import torch
def load_glm_t5(modelcard):
    print('Load the model and tokenizer')
    model = AutoModel.from_pretrained(modelcard, trust_remote_code=True, revision='main')
    tokenizer = AutoTokenizer.from_pretrained(modelcard)
    return model,tokenizer
def g_tokenizer(tokenizer, model, graphs, texts , how):

    how = how  # can be 'global' or 'local', depending on whether the local or global GLM should be used. See paper for more details. 
    data = model.data_processor.encode_graph(tokenizer=tokenizer, g=graphs, text=texts, how=how)
    return data
def graph_to_batch(tokenizer, model,datas):
    model_inputs = model.data_processor.to_batch(data_instances=datas, tokenizer=tokenizer, max_seq_len=1024)
    # print('model.device',model.device)
    # print(len(datas))
    # print(datas)
    return model_inputs
    # device = torch.device("cuda")
    # for key in model_inputs.keys():
    #     model_inputs[key] = model_inputs[key].to(device)
    # print('compute token encodings')
def g_encoder(model, model_inputs):
    # print(model_inputs)
    outputs = model(**model_inputs)
    return outputs
# class GLM_T5(object):
#     def load_glm_t5(modelcard):
#         print('Load the model and tokenizer')
#         model = AutoModel.from_pretrained(modelcard, trust_remote_code=True, revision='main')
#         tokenizer = AutoTokenizer.from_pretrained(modelcard)
#         return model,tokenizer
#     def g_encoder(model, tokenizer, graph, text , how):

#         how = how  # can be 'global' or 'local', depending on whether the local or global GLM should be used. See paper for more details. 
#         data = model.data_processor.encode_graph(tokenizer=tokenizer, g=graph, text=text, how=how)
#         datas = [data]
#         model_inputs = model.data_processor.to_batch(data_instances=datas, tokenizer=tokenizer, max_seq_len=None)

#         print('compute token encodings')
#         outputs = model(**model_inputs)
        
#         return outputs

if __name__ == "__main__":
    modelcard = '/models/glm_t5_large'
    graph =[ 
        ('dog', 'is a', 'animal'),
        ('dog', 'has', 'tail'),
        ('dog', 'has', 'fur'),
        ('fish', 'is a', 'animal'),
        ('fish', 'has', 'scales')]

    # graph=[]
    text = 'summarize: The black poodle chased the cat.' # only graph for this instance
    # glm_t5 = GLM_T5()
    model, tokenizer = load_glm_t5(modelcard)
    data = g_tokenizer(tokenizer,model,graphs = graph ,texts = text, how='global')
    print('type(data)',type(data))
    model_inputs = graph_to_batch(tokenizer,model,[data])
    print('type(model_inputs)',type(model_inputs))
    outputs = g_encoder(model, model_inputs)
    print('prepare model inputs')
    # get token embeddings
    print('Sequence of tokens (batch_size, max_seq_len, embedding_dim):', outputs.last_hidden_state.shape)  # embeddings of all graph and text tokens. Nodes in the graph (e.g., dog) appear only once in the sequence.
    # print('embedding of `dog` in the first instance. Shape is (seq_len, embedding_dim):', model.data_processor.get_embedding(sequence_embedding=outputs.last_hidden_state[0], indices=data.indices, concept='dog', embedding_aggregation='seq').shape)  # embedding_aggregation can be 'seq' or 'mean'. 'seq' returns the sequence of embeddings (e.g., all tokens of `black poodle`), 'mean' returns the mean of the embeddings.
