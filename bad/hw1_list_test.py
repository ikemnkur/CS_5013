from collections import defaultdict, deque

def BFS(start: str, adj_list: dict) -> list:
    queue = deque([start])
    visited = set([start])
    expanded = []

    while queue:
        node = queue.popleft()
        expanded.append(node)

        # Get all neighbors and sort them alphabetically
        neighbors = sorted(adj_list[node])

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return expanded

def create_adjacency_list_from_matrix(csv_content):
    adj_list = defaultdict(list)
    lines = csv_content.strip().split('\n')
    nodes = lines[0].split(',')[1:]  # Get node names, skip the first empty cell
    
    for i, line in enumerate(lines[1:], start=0):
        values = line.split(',')
        node = values[0]
        for j, weight in enumerate(values[1:], start=0):
            if weight != '-1' and weight != '0':
                adj_list[node].append(nodes[j])
    
    return adj_list

def create_adjacency_matrix(adj_list):
    nodes = sorted(adj_list.keys())
    n = len(nodes)
    matrix = [[-1] * n for _ in range(n)]
    
    for i, node in enumerate(nodes):
        matrix[i][i] = 0  # Set diagonal to 0
        for neighbor in adj_list[node]:
            neighbor, weight = neighbor.split(':')
            j = nodes.index(neighbor)
            matrix[i][j] = int(weight)
    
    return nodes, matrix

def matrix_to_string(nodes, matrix):
    header = ',' + ','.join(nodes)
    rows = [header]
    for i, row in enumerate(matrix):
        row_str = f"{nodes[i]}," + ','.join(map(str, row))
        rows.append(row_str)
    return '\n'.join(rows)

# Given adjacency list
adj_list = {
    'A': ['B:4', 'E:1'],
    'B': ['A:4', 'C:2', 'F:2'],
    'C': ['B:2', 'H:4', 'S:3'],
    'D': ['L:8', 'S:2'],
    'E': ['A:1', 'F:3', 'I:6'],
    'F': ['B:2', 'E:3', 'J:6', 'K:4'],
    'G': ['M:4', 'N:4', 'Q:10'],
    'H': ['C:4', 'K:3', 'L:7'],
    'I': ['E:6', 'J:1', 'M:5'],
    'J': ['F:6', 'I:1', 'K:3', 'N:3'],
    'K': ['F:4', 'H:3', 'J:3', 'L:9', 'P:3'],
    'L': ['D:8', 'H:7', 'K:9', 'Q:10'],
    'M': ['G:4', 'I:5'],
    'N': ['G:4', 'J:3', 'P:2'],
    'P': ['K:3', 'N:2'], 
    'Q': ['G:10', 'L:10'],
    'S': ['C:3', 'D:2']
}



# Generate adjacency matrix
nodes, matrix = create_adjacency_matrix(adj_list)

# Convert matrix to CSV string
csv_content = matrix_to_string(nodes, matrix)

print("Generated Adjacency Matrix:")
print(csv_content)

# Test case
# result = BFS('A', adj_list)
# print("\nBFS result starting from 'A':", result)
assert BFS('A') == ['A', 'B', 'E', 'C', 'F', 'I', 'H', 'S', 'J', 'K', 'M', 'G'], "Test case 1 failed"
# assert result == ['A', 'B', 'E', 'C', 'F', 'I', 'H', 'S', 'J', 'M', 'K', 'D', 'N', 'G', 'L', 'P', 'Q'], "Test case failed"
print("Test case passed successfully!")