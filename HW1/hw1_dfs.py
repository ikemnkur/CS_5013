import csv
from collections import defaultdict, deque

# Given adjacency list (it may have errors)
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

# Given adjacency list
adjacency_list = {
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


def create_graph_from_csv(file_content):
    graph = defaultdict(dict)
    reader = csv.reader(file_content.strip().split('\n'))
    nodes = next(reader)[1:]  # Skip the first empty cell
    for i, row in enumerate(reader):
        for j, weight in enumerate(row[1:]):  # Skip the first cell (node name)
            if weight != '-1' and weight != '0':
                graph[row[0]][nodes[j]] = int(weight)
    return graph

def create_adjacency_list_from_matrix(adjacency_matrix):
    adjacency_list = defaultdict(list)
    lines = adjacency_matrix.strip().split('\n')
    nodes = lines[0].split(',')[1:]  # Get node names, skip the first empty cell
    
    for i, line in enumerate(lines[1:], start=0):
        values = line.split(',')
        node = values[0]
        for j, weight in enumerate(values[1:], start=0):
            if weight != '-1' and weight != '0':
                adjacency_list[node].append(nodes[j])
    
    return adjacency_list
    

def create_adjacency_matrix(adjacency_list):
    nodes = sorted(adjacency_list.keys())
    n = len(nodes)
    matrix = [[-1] * n for _ in range(n)]
    
    for i, node in enumerate(nodes):
        matrix[i][i] = 0  # Set diagonal to 0
        for neighbor in adjacency_list[node]:
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


# Implement this Depth First Search function 
def DFS(start: str) -> list:
    graph = create_adjacency_list_from_matrix(adjacency_matrix)
    visited = set()
    expanded = []
    goal_reached = False

    def dfs_recursive(node):
        nonlocal goal_reached
        if node not in visited and not goal_reached:
            visited.add(node)
            expanded.append(node)
            
            if node == 'G':
                goal_reached = True
                return

            # Sort neighbors alphabetically
            neighbors = sorted(graph[node])
            for neighbor in neighbors:
                if neighbor not in visited and not goal_reached:
                    dfs_recursive(neighbor)

    dfs_recursive(start)
    return expanded


# Test cases
def run_tests():
    
    # Test case 3: DFS starting from node 'A'
    result_a = DFS('A')
    print("DFS('A'):", result_a)
    assert DFS('A') == ['A', 'B', 'C', 'H', 'K', 'F', 'E', 'I', 'J', 'N', 'G'], "Test case 3 failed"
    
    # Test case 4: DFS starting from node 'S'
    result_s = DFS('S')
    print("DFS('S'):", result_s)
    assert DFS('S') == ['S', 'C', 'B', 'A', 'E', 'F', 'J', 'I', 'M', 'G'], "Test case 4 failed"


    print("All test cases passed!")




# Generate adjacency list
adj_list = create_adjacency_list_from_matrix(adjacency_matrix)


# Modified function to print adjacency list with weights
def print_adjacency_list(adj_list):
    print("\nGenerated Adjacency List:")
    for node, neighbors in adj_list.items():
        neighbor_strings = [f"{neighbor.split(':')[0]}:{neighbor.split(':')[1]}" for neighbor in neighbors]
        print(f"{node}: {neighbor_strings}")

# Function to check if 2 matrices are equal
def check_adjacency_matrices(matrix1, matrix2):
    if isinstance(matrix1, str) and isinstance(matrix2, str):
        matrix1 = matrix1.strip().split('\n')
        matrix2 = matrix2.strip().split('\n')
    
    if len(matrix1) != len(matrix2):
        return False
    
    for row1, row2 in zip(matrix1, matrix2):
        if row1 != row2:
            return False
    
    return True


# Function to check if 2 adjacency lists are equal
def check_adjacency_lists(list1, list2):
    if len(list1) != len(list2):
        return False
    
    for node in list1:
        if node not in list2:
            return False
        if set(list1[node]) != set(list2[node]):
            return False
    
    return True


# Generate adjacency matrix
nodes, matrix = create_adjacency_matrix(adjacency_list)

# Convert matrix to CSV string
adj_matrix = matrix_to_string(nodes, matrix)

print("Generated Adjacency Matrix:")
print(adj_matrix)

# Generate adjacency list
adj_list = create_adjacency_list_from_matrix(adjacency_matrix)

# Print the adjacency list with weights
print_adjacency_list(adjacency_list)

# Check if adjacency matrices are equal
print("Adj. matrices are equal:", check_adjacency_matrices(adjacency_matrix, adj_matrix))

# Check if adjacency lists are equal
print("Adj. lists are equal:", check_adjacency_lists(adjacency_list, adj_list))

if __name__ == '__main__':
    run_tests()
