import numpy as np
import pandas as pd
import random

def generate_graph_csv(num_nodes, max_weight=10, density=0.5, output_file='graph.csv'):
    """
    Generate a random weighted undirected graph and save it as a CSV file.
    
    Args:
        num_nodes (int): Number of nodes in the graph
        max_weight (int): Maximum weight for edges
        density (float): Probability of an edge existing between any two nodes (0-1)
        output_file (str): Name of the output CSV file
    """
    # Create an empty adjacency matrix
    matrix = np.zeros((num_nodes, num_nodes))
    
    # Generate random weights for edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < density:
                weight = random.randint(1, max_weight)
                matrix[i][j] = weight
                matrix[j][i] = weight  # Make it symmetric for undirected graph
    
    # Create node labels (A, B, C, ...)
    node_labels = [chr(65 + i) for i in range(num_nodes)]
    
    # Convert to DataFrame
    df = pd.DataFrame(matrix, columns=node_labels, index=node_labels)
    
    # Save to CSV
    df.to_csv('data/'+output_file)
    print(output_file)
    print(f"Graph saved to {'data/'+output_file}")

# Example usage
if __name__ == "__main__":
    # Generate a graph with 10 nodes, weights up to 10, and 50% density
    generate_graph_csv(10, 10, 0.5, 'generated_graph.csv')