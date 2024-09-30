import csv
from collections import defaultdict, deque

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

def create_graph_from_csv(file_content):
    graph = defaultdict(dict)
    reader = csv.reader(file_content.strip().split('\n'))
    nodes = next(reader)[1:]  # Skip the first empty cell
    for i, row in enumerate(reader):
        for j, weight in enumerate(row[1:]):  # Skip the first cell (node name)
            if weight != '-1' and weight != '0':
                graph[row[0]][nodes[j]] = int(weight)
    return graph

def BFS(start: str) -> list:
    graph = create_graph_from_csv(csv_content)
    queue = deque([start])
    visited = set([start])
    expanded = []

    while queue:
        node = queue.popleft()
        expanded.append(node)
        if node == 'G':
            break  
        neighbors = sorted(graph[node].keys())
        for neighbor in neighbors:
            if neighbor not in visited and neighbor not in queue:
                queue.append(neighbor)
                visited.add(neighbor)

    return expanded

# Test cases
def run_tests():
    # Test case 1: BFS starting from node 'A'
    result_a = BFS('A')
    print("BFS('A'):", result_a)
    assert result_a == ['A', 'B', 'E', 'C', 'F', 'I', 'H', 'S', 'J', 'K', 'M', 'G'], "Test case 1 failed"
    
    # Test case 2: BFS starting from node 'S'
    result_s = BFS('S')
    print("BFS('S'):", result_s)
    assert result_s == ['S', 'C', 'D', 'B', 'H', 'L', 'A', 'F', 'K', 'Q', 'G'], "Test case 2 failed"

    print("All test cases passed!")

if __name__ == '__main__':
    run_tests()