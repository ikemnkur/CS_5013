import csv
from collections import defaultdict, deque

def create_graph_from_csv(file_content):
    graph = defaultdict(dict)
    reader = csv.reader(file_content.strip().split('\n'))
    
    # Get node names from the first row, skipping the first empty cell
    nodes = next(reader)[1:]
    
    for i, row in enumerate(reader):
        node = row[0]  # Current node is the first column of each row
        for j, weight in enumerate(row[1:]):
            if weight != '-1':
                graph[node][nodes[j]] = int(weight)
    
    return graph

def BFS(start: str) -> list:
    graph = create_graph_from_csv(csv_content)
    queue = deque([start])
    visited = set([start])
    expanded = []

    while queue:
        node = queue.popleft()
        expanded.append(node)

        # Get all neighbors
        neighbors = list(graph[node].keys())
        
        # Sort neighbors alphabetically
        neighbors.sort()

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return expanded

# CSV content as a string
csv_content = """
,A,B,C,D,E,F,G,H,I,J,K,L,M,N,P,Q,S
A,0,4,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
B,4,0,2,-1,-1,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
C,-1,2,0,-1,-1,-1,-1,4,-1,-1,-1,-1,-1,-1,-1,-1,3
D,-1,-1,-1,0,-1,-1,-1,-1,-1,-1,-1,8,-1,-1,-1,-1,2
E,1,-1,-1,-1,0,3,-1,-1,6,-1,-1,-1,-1,-1,-1,-1,-1
F,-1,2,-1,-1,3,0,-1,-1,-1,6,4,-1,-1,-1,-1,-1,-1
G,-1,-1,-1,-1,-1,-1,0,-1,-1,-1,-1,-1,4,4,-1,10,-1
H,-1,-1,4,-1,-1,-1,-1,0,-1,-1,3,7,-1,-1,-1,-1,-1
I,-1,-1,-1,-1,6,-1,-1,-1,0,1,-1,-1,5,-1,-1,-1,-1
J,-1,-1,-1,-1,-1,6,-1,-1,1,0,3,-1,-1,3,-1,-1,-1
K,-1,-1,-1,-1,-1,4,-1,3,-1,3,0,9,-1,-1,3,-1,-1
L,-1,-1,-1,8,-1,-1,-1,7,-1,-1,9,0,-1,-1,-1,10,-1
M,-1,-1,-1,-1,-1,-1,4,-1,5,-1,-1,-1,0,-1,-1,-1,-1
N,-1,-1,-1,-1,-1,-1,4,-1,-1,3,-1,-1,-1,0,2,-1,-1
P,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,3,-1,-1,2,0,-1,-1
Q,-1,-1,-1,-1,-1,-1,10,-1,-1,-1,-1,10,-1,-1,-1,0,-1
S,-1,-1,3,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0
"""

# Test case
result = BFS('A')
print(result)

# Test case 1: BFS starting from node 'A'
assert BFS('A') == ['A', 'B', 'E', 'C', 'F', 'I', 'H', 'S', 'J', 'K', 'M', 'G'], "Test case 1A failed"
    
assert result == ['A', 'B', 'E', 'C', 'F', 'I', 'H', 'S', 'J', 'K', 'M', 'G'], "Test case 1B failed"
print("Test case passed successfully!")