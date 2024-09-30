from collections import defaultdict, deque

def create_adjacency_list(file_content):
    adj_list = defaultdict(list)
    lines = file_content.strip().split('\n')
    nodes = lines[0].split(',')[1:]  # Get node names, skip the first empty cell
    
    for i, line in enumerate(lines[1:], start=0):
        values = line.split(',')
        node = values[0]
        for j, weight in enumerate(values[1:], start=0):
            if weight != '-1' and weight != '0':
                adj_list[node].append(nodes[j])
    
    return adj_list

def BFS(start: str) -> list:
    graph = create_adjacency_list(csv_content)
    queue = deque([start])
    visited = set([start])
    expanded = []

    while queue:
        node = queue.popleft()
        expanded.append(node)

        # Get all neighbors and sort them alphabetically
        neighbors = sorted(graph[node])

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return expanded

# CSV content as a string
csv_content = """
,A,B,C,D,E,F,G,H,I,J,K,L,M,N,P,Q,S
A,0,1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
B,1,0,1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
C,-1,1,0,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,1
D,-1,-1,-1,0,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1
E,1,-1,-1,-1,0,1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1
F,-1,1,-1,-1,1,0,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1
G,-1,-1,-1,-1,-1,-1,0,-1,-1,-1,-1,-1,1,1,-1,1,-1
H,-1,-1,1,-1,-1,-1,-1,0,-1,-1,1,1,-1,-1,-1,-1,-1
I,-1,-1,-1,-1,1,-1,-1,-1,0,1,-1,-1,1,-1,-1,-1,-1
J,-1,-1,-1,-1,-1,1,-1,-1,1,0,1,-1,-1,1,-1,-1,-1
K,-1,-1,-1,-1,-1,1,-1,1,-1,1,0,1,-1,-1,1,-1,-1
L,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,0,-1,-1,-1,1,-1
M,-1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,0,-1,-1,-1,-1
N,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,0,1,-1,-1
P,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,0,-1,-1
Q,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,0,-1
S,-1,-1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0
"""

# Print the adjacency list for verification
print("\nAdjacency List:")
adj_list = create_adjacency_list(csv_content)
for node, neighbors in adj_list.items():
    print(f"{node}: {neighbors}")

# Test case
result = BFS('A')
print(result)

assert result == ['A', 'B', 'E', 'C', 'F', 'I', 'H', 'S', 'J', 'K', 'M', 'G'], "Test case 1 failed"
# assert BFS('A') == ['A', 'B', 'E', 'C', 'F', 'I', 'H', 'S', 'J', 'K', 'M', 'G'], "Test case 1 failed"
    
print("Test case passed successfully!")

