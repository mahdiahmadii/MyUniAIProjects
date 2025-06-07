import pandas as pd
import numpy as np

def get_matrix_from_user():
    """
    Get matrix input from user through console.
    """
    print("Enter the number of nodes in the graph:")
    n = int(input())
    
    print(f"Enter the {n}x{n} matrix (one row at a time, space-separated values):")
    matrix = []
    for i in range(n):
        row = list(map(int, input().split()))
        if len(row) != n:
            raise ValueError(f"Expected {n} values in row {i+1}")
        matrix.append(row)
    
    return matrix

def matrix_to_csv(matrix, output_file='graph_input.csv'):
    """
    Convert a matrix to CSV format with node labels.
    """
    # Convert to numpy array
    matrix = np.array(matrix)
    
    # Generate node labels (A, B, C, ...)
    node_labels = [chr(65 + i) for i in range(len(matrix))]
    
    # Create DataFrame
    df = pd.DataFrame(matrix, columns=node_labels, index=node_labels)
    
    # Save to CSV
    df.to_csv('data/'+output_file)
    print(f"Graph saved to {'data/'+output_file}")

def main():
    try:
        # Get matrix from user
        matrix = get_matrix_from_user()
        
        # Convert to CSV
        matrix_to_csv(matrix)
        print("Conversion successful!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()