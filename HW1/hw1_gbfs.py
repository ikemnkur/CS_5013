from collections import defaultdict
import heapq

# Given heuristic function
H_n = {
    "s": 17, "a": 10, "b": 9, "c": 16, "d": 21, "e": 13, "f": 9, "g": 0,
    "h": 12, "i": 9, "j": 5, "k": 8, "l": 18, "m": 3, "n": 4, "p": 6, "q": 9
}

def create_adjacency_list_from_matrix(adjacency_matrix):
    adjacency_list = defaultdict(list)
    lines = adjacency_matrix.strip().split('\n')
    nodes = lines[0].split(',')[1:]  # Get node names, skip the first empty cell
    
    for i, line in enumerate(lines[1:], start=0):
        values = line.split(',')
        node = values[0].lower()  # Convert to lowercase
        for j, weight in enumerate(values[1:], start=0):
            if weight != '-1' and weight != '0':
                adjacency_list[node].append(f"{nodes[j].lower()}:{weight}")
    
    return adjacency_list

def heuristic(node):
    return H_n.get(node.lower(), float('inf'))

def GBFS(start: str) -> list:
    graph = create_adjacency_list_from_matrix(adjacency_matrix)
    visited = set()
    expanded = []
    pq = [(heuristic(start.lower()), start.lower())]

    while pq:
        _, node = heapq.heappop(pq)
        if node not in visited:
            visited.add(node)
            expanded.append(node.upper())  # Convert back to uppercase for output

            if node == 'g':
                break

            neighbors = [n.split(':')[0] for n in graph[node]]
            for neighbor in neighbors:
                if neighbor not in visited:
                    heapq.heappush(pq, (heuristic(neighbor), neighbor))

    return expanded

# Adjacency matrix
adjacency_matrix = """
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

# Test cases
def run_tests():
    # Test case 5: GBFS starting from node 'A'
    result_a = GBFS('A')
    print("GBFS('A'):", result_a)
    assert result_a == ['A', 'B', 'F', 'J', 'N', 'G'], "Test case 5 failed"
    
    # Test case 6: GBFS starting from node 'S'
    result_s = GBFS('S')
    print("GBFS('S'):", result_s)
    assert result_s == ['S', 'C', 'B', 'F', 'J', 'N', 'G'], "Test case 6 failed"

    # Test case 5: GBFS starting from node 'A'
    assert GBFS('A') == ['A', 'B', 'F', 'J', 'N', 'G'], "Test case 5 failed"
    
    # Test case 6: GBFS starting from node 'S'
    assert GBFS('S') == ['S', 'C', 'B', 'F', 'J', 'N', 'G'], "Test case 6 failed"

    

    print("All GBFS test cases passed!")

if __name__ == '__main__':
    run_tests()