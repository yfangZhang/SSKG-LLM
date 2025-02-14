import networkx as nx
import random
import pandas as pd
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
                edge_info.append(str(vertex)+' '+str(relation)+' '+str(neighbor))
                # 如果相邻节点未被访问过，则将其压入栈中
                if neighbor not in visited:
                    stack.append(neighbor)
            # # 获取相邻节点并压入栈中
            # stack.extend(reversed(list(graph.neighbors(vertex))))
    return edge_info
def dfs_k_relations(graph, start, K):
    visited = set()
    stack = [start]
    edge_info = []
    count = 0  # 计数器

    while stack and count < K:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            for neighbor in graph.neighbors(vertex):
                # 获取当前节点和相邻节点之间的关系类型
                relation = graph.edges[vertex, neighbor]['relation']
                # 将关系类型添加到edge_info字典中
                edge_info.append(f"{vertex} {relation} {neighbor}")
                count += 1  # 增加计数
                # 如果相邻节点未被访问过，则将其压入栈中
                if neighbor not in visited:
                    stack.append(neighbor)
                # 如果已获取K组关系，退出循环
                if count >= K:
                    break
        # 如果已获取K组关系，退出循环
        if count >= K:
            break

    return edge_info

def get_structure_tokens(relations):
    # 创建一个简单的无向图
    # G = nx.Graph()
    # 创建一个简单的有向图
    G = nx.DiGraph()
    rel_set = []
    source = []
    target = []
    rel = []
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
    start_node = random.choice(relations)
    # print('start_node',start_node)
    start_node = start_node['head_entity']
    # return dfs(G, start_node)
    return dfs_k_relations(G,start_node,K=5)
    # walk_path, visited_nodes = random_walk(G, start_node)
    # print("随机游走路径:", walk_path)
    # print("访问的节点:", visited_nodes)
if __name__ == "__main__":
    # question = "hello,i love you"
    # # sub_kgs = get_relations(question)
    
    # path = get_structure_tokens(sub_kgs)
    # print(type(path))
    print()
