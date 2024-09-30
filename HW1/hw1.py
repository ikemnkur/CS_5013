# Problem: Implement the Breadth-First Search (BFS), Depth-First Search (DFS) 
# and Greedy Best-First Search (GBFS) algorithms on the graph from Figure 1 in hw1.pdf.


# Instructions:
# 1. Represent the graph from Figure 1 in any format (e.g. adjacency matrix, adjacency list).
# 2. Each function should take in the starting node as a string. Assume the search is being performed on
#    the graph from Figure 1.
#    It should return a list of all node labels (strings) that were expanded in the order they where expanded.
#    If there is a tie for which node is expanded next, expand the one that comes first in the alphabet.
# 3. You should only modify the graph representation and the function body below where indicated.
# 4. Do not modify the function signature or provided test cases. You may add helper functions. 
# 5. Upload the completed homework to Gradescope, it must be named 'hw1.py'.

# Examples:
#     The test cases below call each search function on node 'S' and node 'A'
# -----------------------------


# def BFS(start: str) -> list:
#     # START: Your code here
#     return []
#     # END: Your code here

import csv
from collections import defaultdict, deque
import heapq

# CSV content as a string

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

# Given heuristic function
H_n = {
    "S": 17, "A": 10, "B": 9, "C": 16, "D": 21, "E": 13, "F": 9, "G": 0,
    "H": 12, "I": 9, "J": 5, "K": 8, "L": 18, "M": 3, "N": 4, "K": 6, "Q": 9
}

# # Given adjacency list
# astar_adjacency_list = {
#     'H': ['c:4', 'E:3', 'F:2'],
#     'C': ['A:2', 'D:3', 'E:3'],
#     'D': ['C:3', 'A:2', 'K:2'],
#     'K': ['G:12', 'D:2', 'J:2'],
#     'E': ['A:5', 'F:6', 'H:3', 'C:3'],
#     'F': ['H:2', 'E:6', 'B:5', 'G:7'],
#     'G': ['J:4', 'B:3', 'A:10', 'J:4', 'K:12' ],
#     'J': ['G:4', 'K:2', 'B:4'],
#     'A': ['C:2', 'D:4', 'E:5', 'G:10'],
#     'N': ['F:5', 'G:3', 'J:4'],
# }

# The heuristic function
def heuristic(node):
    return H_n.get(node, float('inf'))

# Creates graphs from the csv, which is then use to create a adjacency list (not used, old)
def create_graph_from_csv(file_content):
    graph = defaultdict(dict)
    reader = csv.reader(file_content.strip().split('\n'))
    nodes = next(reader)[1:]  # Skip the first empty cell
    for i, row in enumerate(reader):
        for j, weight in enumerate(row[1:]):  # Skip the first cell (node name)
            if weight != '-1' and weight != '0':
                graph[row[0]][nodes[j]] = int(weight)
    return graph

# Creates adjacency list from the Matrix CSV
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

# Creates adjacency list from the Matrix but considers the Wieghts when appending to the list
def create_adjacency_list_from_matrix_gbfs(adjacency_matrix):
    adjacency_list = defaultdict(list)
    lines = adjacency_matrix.strip().split('\n')
    nodes = lines[0].split(',')[1:]  # Get node names, skip the first empty cell
    
    for i, line in enumerate(lines[1:], start=0):
        values = line.split(',')
        node = values[0]
        for j, weight in enumerate(values[1:], start=0):
            if weight != '-1' and weight != '0':
                adjacency_list[node].append(f"{nodes[j]}:{weight}")
    
    return adjacency_list

# Creates an Adjaceny Matrix for comparing the inputs
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

# Convert Matrix Object in to String to print to console
def matrix_to_string(nodes, matrix):
    header = ',' + ','.join(nodes)
    rows = [header]
    for i, row in enumerate(matrix):
        row_str = f"{nodes[i]}," + ','.join(map(str, row))
        rows.append(row_str)
    return '\n'.join(rows)



# Implement this Breath First Search function 
def BFS(start: str) -> list:
    graph = create_adjacency_list_from_matrix(adjacency_matrix)
    queue = deque([start])
    visited = set([start])
    expanded = []

    while queue:
        node = queue.popleft()
        expanded.append(node)

        if node == 'G':
            break  

        # Get all neighbors and sort them alphabetically
        neighbors = sorted(graph[node])

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
            if neighbor == 'G':
                print("Node: ", node, " has nieghbor node G")
                expanded.append(neighbor)
                return expanded

    return expanded

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

# Implement this Greedy Best First Search function 
def GBFS(start: str) -> list:
    graph = create_adjacency_list_from_matrix(adjacency_matrix)
    visited = set()
    expanded = []
    pq = [(heuristic(start), start)]

    while pq:
        unused, node = heapq.heappop(pq)
        if node not in visited:
            visited.add(node)
            expanded.append(node)

            if node == 'G':
                break

            neighbors = [n.split(':')[0] for n in graph[node]]
            for neighbor in neighbors:
                if neighbor not in visited:
                    heapq.heappush(pq, (heuristic(neighbor), neighbor))

    return expanded

#Print BFS Test cases
print("BFS('A'):", BFS('A'))
print("BFS('S'):", BFS('S'))

# Print DFS Test cases
print("DFS('A'):", DFS('A'))
print("DFS('S'):", DFS('S'))
 
# Print GBFS Test cases 
print("GBFS('A'):", GBFS('A'))
print("GBFS('S'):", GBFS('S'))




# Generate adjacency list
adj_list = create_adjacency_list_from_matrix(adjacency_matrix)

# Modified function to print adjacency list with weights
def print_adjacency_list(adj_list):
    print("\nGenerated Adjacency List:")
    for node, neighbors in adj_list.items():
        neighbor_strings = [f"{neighbor.split(':')[0]}:{neighbor.split(':')[1]}" for neighbor in neighbors]
        print(f"{node}: {neighbor_strings}")

# Old Function to print adjacency list (no wieghts)
def print_gen_adjacency_list(adj_list):
    print("\nGenerated Adjacency List:")
    for node, neighbors in adj_list.items():
        print(f"{node}: {neighbors}")

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

# print Matrix CSV
print("Generated Adjacency Matrix:")
print(adj_matrix)

# Generate adjacency list
adj_list = create_adjacency_list_from_matrix(adjacency_matrix)

# Print the old adjacency list with weights
print_adjacency_list(adjacency_list)

# Print the new adjacency list with weights
print_gen_adjacency_list(adj_list)

# Print the new adjacency list with weights with the python print function
print("\nRaw Adjacency List:")
print(dict(adj_list))

print("\n") # for spacing

# Check if adjacency matrices are equal
print("Adj. matrices are equal:", check_adjacency_matrices(adjacency_matrix, adj_matrix))

# Check if adjacency lists are equal - not working but they are the same, I check manually
print("Adj. lists are equal:", check_adjacency_lists(adjacency_list, adj_list))



# test cases - DO NOT MODIFY THESE
def run_tests():
    # Test case 1: BFS starting from node 'A'
    assert BFS('A') == ['A', 'B', 'E', 'C', 'F', 'I', 'H', 'S', 'J', 'K', 'M', 'G'], "Test case 1 failed"
    
    # Test case 2: BFS starting from node 'S'
    assert BFS('S') == ['S', 'C', 'D', 'B', 'H', 'L', 'A', 'F', 'K', 'Q', 'G'], "Test case 2 failed"

    # Test case 3: DFS starting from node 'A'
    assert DFS('A') == ['A', 'B', 'C', 'H', 'K', 'F', 'E', 'I', 'J', 'N', 'G'], "Test case 3 failed"
    
    # Test case 4: DFS starting from node 'S'
    assert DFS('S') == ['S', 'C', 'B', 'A', 'E', 'F', 'J', 'I', 'M', 'G'], "Test case 4 failed"

    # Test case 5: GBFS starting from node 'A'
    assert GBFS('A') == ['A', 'B', 'F', 'J', 'N', 'G'], "Test case 5 failed"
    
    # Test case 6: GBFS starting from node 'S'
    assert GBFS('S') == ['S', 'C', 'B', 'F', 'J', 'N', 'G'], "Test case 6 failed"
    
    print("All test cases passed!")

if __name__ == '__main__':
    run_tests()
