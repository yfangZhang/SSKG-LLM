import networkx as nx
import random,string
import pandas as pd
index = 0
# from get_subkgs import get_relations
def random_walk(graph, start_node):
    current_node = start_node
    visited = {current_node}  # 记录已访问的节点
    path = [current_node]

    while len(visited) < len(graph.nodes):  # 当未访问的节点小于总节点数时继续
        neighbors = list(graph.successors(current_node))
        if not neighbors:  # 如果没有邻居，则结束游走
            break
        current_node = random.choice(neighbors)  # 随机选择一个邻居
        path.append(current_node)
        visited.add(current_node)  # 添加当前节点到已访问集合
    
    return path, visited
import networkx as nx
def dfs(graph, start):
    visited = set()
    stack = [start]
    edge_info = []
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            # print(vertex)
            for neighbor in graph.neighbors(vertex):
                # 获取当前节点和相邻节点之间的关系类型
                relation = graph.edges[vertex, neighbor]['relation']
                # 将关系类型添加到edge_info字典中
                edge_info.append(str(vertex)+' '+str(relation))
                # 如果相邻节点未被访问过，则将其压入栈中
                if neighbor not in visited:
                    stack.append(neighbor)
            # # 获取相邻节点并压入栈中
            # stack.extend(reversed(list(graph.neighbors(vertex))))
    return edge_info
def my_visit(graph,node):
    other_relations = []
    result = []
    # result.append(node)
    try:
        for neighbor, edge_data in graph[node].items():
            relation = edge_data['relation']
            other_relations.append((neighbor, relation))
        for neighbor, relation in other_relations:
            result.append(f"{relation} {neighbor}")
                # dfs_visit(neighbor)

        # 添加<KG_END>标记
        # print("<KG_END>")
        result.append("<KG_END>")
    except:
        pass
    return result
def my_dfs(graph, start, visited=None, result=None, max_hops=3, max_nodes=1):
    if visited is None:
        visited = set()
    if result is None:
        result = []
    
    def dfs_visit(node, current_hop, max_nodes=1):
        if node not in visited:
            visited.add(node)
            print((current_hop,max_hops))
            if current_hop < max_hops:
                other_relations = []
                try:
                    neighbor_count = 0
                    for neighbor, edge_data in graph[node].items():
                        if neighbor_count < max_nodes:
                            relation = edge_data['relation']
                            result.append(f"{relation} {neighbor}")
                            dfs_visit(neighbor, current_hop + 1, max_nodes)
                        else:
                            break
                    
                except:
                    pass

    dfs_visit(start, 0, max_nodes)
    if len(result)>0:
        result.append("<KG_END>")
        result.insert(0,"<KG_START>")
    return result
from collections import deque

def bfs_k_hops(graph, start_node, max_hops, max_nodes):
    visited = set()
    queue = deque([(start_node, 0)])  # (node, current_hops)
    result = []

    while queue and len(result) < max_nodes:
        current_node, current_hops = queue.popleft()

        if current_hops > max_hops:
            continue
        try:
            for neighbor, edge_data in graph[start_node].items():
                if neighbor not in visited:
                    visited.add(neighbor)
                    if current_hops < max_hops:
                        queue.append((neighbor, current_hops + 1))
                        relation = edge_data['relation']
                        result.append(f"{relation} {neighbor}")
                        if len(result) == max_nodes:
                            break
        except:
            pass
    if len(result)>0:
        result.append("<KG_END>")
        result.insert(0,"<KG_START>")
    return result
with open('/qagnn-main/data/cpnet/concept.txt', "r", encoding="utf8") as fin:
    id2concept = [w.strip() for w in fin]
concept2id = {w: i for i, w in enumerate(id2concept)}
def bfs_k_hops_prekgs(graph, start_node, max_hops, max_nodes):
    visited = set()
    queue = deque([(start_node, 0)])  # (node, current_hops)
    result = []

    while queue and len(result) < max_nodes:
        current_node, current_hops = queue.popleft()

        if current_hops > max_hops:
            continue
        try:
            for neighbor, edge_data in graph[start_node].items():
                if neighbor not in visited:
                    visited.add(neighbor)
                    if current_hops < max_hops:
                        queue.append((neighbor, current_hops + 1))
                        relation = edge_data['relation']
                        # result.append(f"{concept2id[current_node]} {concept2id[relation]} {concept2id[neighbor]}")
                        result.append(f"{current_node} {relation} {neighbor}")
                        if len(result) == max_nodes:
                            break
        except:
            pass
    return result

def get_structure_tokens(relations,text):
    # 创建一个简单的无向图
    # G = nx.Graph()
    # 创建一个简单的有向图
    G = nx.DiGraph()
    rel_set = []
    source = []
    target = []
    rel = []
    text_list = text.split(' ')
    text_list = [s for s in text_list if s.strip()]
    trans_table = str.maketrans('', '', string.punctuation)

    # 使用列表推导式和translate方法去除标点符号
    text_list = [s.translate(trans_table) for s in text_list]
    # print(text_list)
    # text_list = [{'head_entity':text_list[i], 'rel':'<next_token>', 'tail_entity':text_list[i+1]} for i in range(len(text_list) - 1)]
    # relations = text_list+relations
    for relation in relations:
        # rel_set.append((relation['head_entity'],relation['rel']))
        # rel_set.append((relation['rel'],relation['tail_entity']))
        source.append(relation['head_entity'])
        target.append(relation['tail_entity'])
        rel.append(relation['rel'])
        rel_set.append((relation['head_entity'],relation['tail_entity']))
    data = {
    'source': source,
    'target': target,
    'relation': rel}
    df = pd.DataFrame(data)

    # 从DataFrame创建图
    G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr=['relation'])
    # G.add_edges_from(rel_set)
    # start_node = random.choice(relations)
    # print('start_node',start_node)
    # start_node = start_node['head_entity']
    result = []
    kgs = []
    for node in text_list:
        result.append(node)
    # start_node = text_list[0]['head_entity']
        # result += my_dfs(G, node,max_hops=1,max_nodes=1)
        kgs += bfs_k_hops_prekgs(G, node,max_hops=1,max_nodes=1)
        result += bfs_k_hops(G, node,max_hops=1,max_nodes=1)
        result.append('<next_token>')

    # walk_path, visited_nodes = random_walk(G, start_node)
    # print("随机游走路径:", walk_path)
    # print("访问的节点:", visited_nodes)
    return result,kgs
if __name__ == "__main__":
    # question = "hello,i love you"
    # # sub_kgs = get_relations(question)
    
    # path = get_structure_tokens(sub_kgs)
    # print(type(path))
    print()
