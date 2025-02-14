from utils.graph import generate_adj_data_from_grounded_concepts
grounded_path = '/my_grpah_llm/data/SIQA_ground.json'
cpnet_graph_path = '/data/cpnet/conceptnet.en.pruned.graph'
cpnet_vocab_path = '/qagnn-main/data/cpnet/concept.txt'
output_path = '../mySIQA_grpah.json'
num_processes=16
generate_adj_data_from_grounded_concepts(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes)