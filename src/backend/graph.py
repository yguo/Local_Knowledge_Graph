import networkx as nx
import heapq

class KnowledgeGraph:
    def __init__(self):
        self.graph_data = {
            'nodes': [],
            'edges': []
        }
        self.edge_dict = {}

    def add_node(self, node_id, label, value=20):
        self.graph_data['nodes'].append({
            'id': node_id,
            'label': label,
            'value': value
        })

    def add_edge(self, from_node, to_node, value, length):
        edge_key = f"{from_node}-{to_node}"
        self.edge_dict[edge_key] = {
            'from': from_node,
            'to': to_node,
            'value': value,
            'length': length
        }
        self.graph_data['edges'] = list(self.edge_dict.values())

    def serialize_graph_data(self):
        serialized = {
            'nodes': self.graph_data['nodes'],
            'edges': [
                {
                    'from': edge['from'],
                    'to': edge['to'],
                    'value': float(edge['value']),
                    'label': f"{float(edge['value']):.2f}",
                    'font': {'size': 10}
                }
                for edge in self.graph_data['edges']
            ]
        }
        return serialized

    def calculate_strongest_path(self, current_step):
        G = nx.Graph()
        for node in self.graph_data['nodes']:
            G.add_node(node['id'])
        for edge in self.graph_data['edges']:
            G.add_edge(edge['from'], edge['to'], weight=edge['value'])

        start_node = 'Step1'
        end_node = f'Step{current_step}'

        def dijkstra(graph, start, end):
            queue = [(0, start, [])]
            visited = set()

            while queue:
                (cost, node, path) = heapq.heappop(queue)
                if node not in visited:
                    visited.add(node)
                    path = path + [node]

                    if node == end:
                        path_length = len(path) - 1
                        if path_length == 0:
                            return 1.0, path
                        return -cost / path_length, path

                    for neighbor in graph.neighbors(node):
                        if neighbor not in visited:
                            edge_weight = graph[node][neighbor]['weight']
                            new_cost = cost - edge_weight
                            heapq.heappush(queue, (new_cost, neighbor, path))

            return None, None

        try:
            avg_similarity, path = dijkstra(G, start_node, end_node)
            if path:
                if len(path) == 1:
                    return path, [], 1.0
                path_edges = list(zip(path[:-1], path[1:]))
                path_weights = [G[u][v]['weight'] for u, v in path_edges]
                return path, path_weights, avg_similarity
            else:
                return None, None, None
        except nx.NetworkXNoPath:
            return None, None, None