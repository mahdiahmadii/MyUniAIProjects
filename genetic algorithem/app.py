
##############################################################################LIBRARIES
import pandas as pd
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

##############################################################################DATA_CLEANING
def read_graph_to_dict(file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, index_col=0)
    
    # Initialize the result dictionary
    distances = {}
    
    # Iterate through each row (source city)
    for src_city in df.index:
        distances[src_city] = {}
        # Iterate through each column (destination city)
        for dst_city in df.columns:
            # Get the distance value
            distance = df.loc[src_city, dst_city]
            # Add to nested dictionary
            distances[src_city][dst_city] = distance
    
    return distances

#-----------------------------------------------------------------------------
def convert_dict_to_matrix(distances_dict):
    """Convert dictionary format to adjacency matrix"""
    nodes = list(distances_dict.keys())
    n = len(nodes)
    matrix = [[0] * n for _ in range(n)]
    
    # Create a mapping of node names to indices
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Fill the matrix
    for node1 in distances_dict:
        for node2 in distances_dict[node1]:
            matrix[node_to_idx[node1]][node_to_idx[node2]] = distances_dict[node1][node2]
    
    return matrix, nodes
##############################################################################INITIALIZING_FIRST_GEN 
def initial_chromosomes():
    return [random.sample(CITIES, len(CITIES)) for _ in range(POP_SIZE)]
##############################################################################PATH_DISTANCE

def path_distance(path):
    total = 0
    for i in range(len(path)):
        src = path[i]
        dst = path[(i + 1) % len(path)]  
        dist = DISTANCES[src][dst]
        if dist == 0:
            return 0
        total += dist
    return total
#-------------------------------------------------------------------------------
def chromosome_measure(chromosomes):
    scores = []
    for chrom in chromosomes:
        if path_distance(chrom) == 0:
            scores.append(0.000001)
        else:
            scores.append(1 / path_distance(chrom))
    return scores
##############################################################################
def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title='Select Input Graph File',
        filetypes=[('CSV Files', '*.csv')],
        initialdir='data/'
    )
    return file_path if file_path else 'data/graph.csv'
##############################################################################
def fitness_population_selection(chromosomes, scores):
    probs = np.array(scores) / sum(scores)
    selected = np.random.choice(len(chromosomes), size=POP_SIZE, p=probs)
    return [chromosomes[i] for i in selected]

#-----------------------------------------------------------------------------
def greedy_selection(chromosomes, scores):
    # Sort chromosomes by their scores in descending order
    sorted_pairs = sorted(zip(chromosomes, scores), 
                         key=lambda x: x[1], 
                         reverse=True)
    
    # Take the top POP_SIZE chromosomes
    selected = [pair[0] for pair in sorted_pairs[:POP_SIZE]]
    
    return selected
##############################################################################
def cross_over(population):
    new_population = []
    for _ in range(0, POP_SIZE, 2):
        parent1, parent2 = random.sample(population, 2)
        child1 = crossover_monoposition(parent1, parent2)
        child2 = crossover_monoposition(parent2, parent1)
        new_population.extend([child1, child2])
    return new_population
#-----------------------------------------------------------------------------
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None]*size

    #Copy slice from parent1
    child[start:end] = parent1[start:end]

    #Fill the rest from parent2
    fill_pos = end
    for city in parent2:
        if city not in child:
            if fill_pos >= size:
                fill_pos = 0
            child[fill_pos] = city
            fill_pos += 1

    return child
#-----------------------------------------------------------------------------
def crossover_monoposition(parent1,parent2):
    size = len(parent1)
    # Choose a single random crossover point
    crossover_point = random.randint(1, size-1)
    
    # Initialize child with None
    child = [None] * size
    
    # Copy first part from parent1 (up to crossover point)
    child[:crossover_point] = parent1[:crossover_point]
    
    # Fill the rest from parent2 (maintaining order and avoiding duplicates)
    remaining_cities = [city for city in parent2 if city not in child]
    child[crossover_point:] = remaining_cities
    
    return child
#-----------------------------------------------------------------------------
def crossover_doubleposition(parent1, parent2):
    size = len(parent1)
    # Choose two random distinct crossover points
    point1, point2 = sorted(random.sample(range(size), 2))
    
    # Initialize child with None
    child = [None] * size
    
    # Copy middle segment from parent1 (between point1 and point2)
    child[point1:point2] = parent1[point1:point2]
    
    # Fill the remaining positions from parent2 (maintaining order and avoiding duplicates)
    # Create a list of cities from parent2 that aren't in the middle segment
    remaining_cities = [city for city in parent2 if city not in child]
    
    # Fill the end part (after point2)
    end_size = size - point2
    child[point2:] = remaining_cities[:end_size]
    
    # Fill the start part (before point1)
    child[:point1] = remaining_cities[end_size:]
    
    return child

##############################################################################
def mutation(population):
    for chrom in population:
        if random.random() < MUTATION_RATE:
            i, j = random.sample(range(len(chrom)), 2)
            chrom[i], chrom[j] = chrom[j], chrom[i]
    return population
#------------------------------------------------------------------------------

def mutation_doublepoint(population, distances):
    for chrom in population:
        if random.random() < MUTATION_RATE:
            i ,j, k , l = random.sample(range(len(chrom)),4)
            chrom[i],chrom[j],chrom[k],chrom[l] = chrom[l],chrom[k],chrom[j],chrom[i]
        return population
    

 

##############################################################################
def max_score(scores):
    return max(scores)
##############################################################################

def totalCost(cost, visited, currPos, n, count, costSoFar, ans, path, current_path):
    """Calculate total cost with path tracking"""
    if count == n and cost[currPos][0] != 0:
        if costSoFar + cost[currPos][0] < ans[0]:
            ans[0] = costSoFar + cost[currPos][0]
            # Save the complete path
            path[0] = current_path + [0]
        return

    for i in range(n):
        if not visited[i] and cost[currPos][i] != 0:
            visited[i] = True
            current_path.append(i)
            totalCost(cost, visited, i, n, count + 1,
                     costSoFar + cost[currPos][i], ans, path, current_path)
            current_path.pop()
            visited[i] = False

def tsp_backtracking(distances_dict):
    # Convert dictionary format to matrix
    cost, nodes = convert_dict_to_matrix(distances_dict)
    n = len(cost)
    visited = [False] * n
    visited[0] = True
    
    ans = [float('inf')]  # For minimum cost
    path = [[]]  # For storing the optimal path
    current_path = [0]  # Start with node 0
    
    totalCost(cost, visited, 0, n, 1, 0, ans, path, current_path)
    
    # Convert numeric path back to node names
    final_path = [nodes[i] for i in path[0]]
    
    return ans[0], final_path
##############################################################################VISUALIZER

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph_from_csv(csv_file_path):
    # Read CSV file
    df = pd.read_csv(csv_file_path, index_col=0)
    
    # Create a new directed graph
    G = nx.DiGraph()
    
    # Add edges with weights
    for source in df.index:
        for target in df.columns:
            weight = df.loc[source, target]
            if weight != 0:  # Only add edges with non-zero weights
                G.add_edge(source, target, weight=weight)
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Create a layout for the graph
    # You can try different layouts:
    # pos = nx.spring_layout(G)
    # pos = nx.circular_layout(G)
    pos = nx.kamada_kawai_layout(G)
    
    # Draw the graph
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue',
                          node_size=500)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos)
    
    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, 
                                edge_labels=edge_labels)
    
    # Remove axes
    plt.axis('off')
    
    # Add title
    plt.title("Graph Visualization", pad=20, size=16)
    
    # Show the plot
    plt.show()
    
    return G

# Alternative version with more customization options
def visualize_graph_from_csv_advanced(csv_file_path, 
                                    node_color='lightblue',
                                    node_size=500,
                                    edge_color='gray',
                                    font_size=10,
                                    layout_type='kamada_kawai',
                                    fig_size=(12, 8)):
    # Read CSV file
    df = pd.read_csv(csv_file_path, index_col=0)
    
    # Create a new directed graph
    G = nx.DiGraph()
    
    # Add edges with weights
    for source in df.index:
        for target in df.columns:
            weight = df.loc[source, target]
            if weight != 0:
                G.add_edge(source, target, weight=weight)
    
    # Set up the plot
    plt.figure(figsize=fig_size)
    
    # Choose layout
    if layout_type == 'spring':
        pos = nx.spring_layout(G)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_color,
                          node_size=node_size)
    
    nx.draw_networkx_edges(G, pos,
                          edge_color=edge_color,
                          arrows=True,
                          arrowsize=20)
    
    nx.draw_networkx_labels(G, pos, font_size=font_size)
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos,
                                edge_labels=edge_labels,
                                font_size=font_size)
    
    plt.axis('off')
    plt.title("Graph Visualization", pad=20, size=16)
    plt.show()
    
    return G

def visualize_graph(data_path):
    
    # G = visualize_graph_from_csv(data_path)
    
    # Advanced usage with customization
    G = visualize_graph_from_csv_advanced(
        data_path,
        node_color='lightgreen',
        node_size=600,
        edge_color='blue',
        font_size=12,
        layout_type='circular',
        fig_size=(10, 10)
    )
   

##############################################################################STARTER
def main():
    visualize_graph(data_path)
    
    start_time = time.time()
    chromosomes = initial_chromosomes()
    initial_chromosomes1 = chromosomes
    scores = chromosome_measure(chromosomes)
    generation = 0
    all_time_best_find = 1000000
    while max_score(scores) < EXPECTED_ERR_PERC / 100 * (1 / MAX_OPTIMUM_SCORE) and len(chromosomes)==len(initial_chromosomes1):
        # chromosomes = fitness_population_selection(chromosomes, scores)
        chromosomes = greedy_selection(chromosomes,scores)
        chromosomes = cross_over(chromosomes)
        chromosomes = mutation(chromosomes)
        # chromosomes = mutation_doublepoint(chromosomes,DISTANCES)
        scores = chromosome_measure(chromosomes)
        generation += 1
        if(1/max_score(scores) < all_time_best_find):
            all_time_best_find = 1/max_score(scores)
            best_index = scores.index(max_score(scores))
            best_route_found =  (" -> ".join(chromosomes[best_index]) + f" -> {chromosomes[best_index][0]}")
        print(f"Generation {generation}: Best score {max_score(scores):.5f}, Best distance {1/max_score(scores):.2f}, all time best = {all_time_best_find}")
    print("____________________________________")
    best_index = scores.index(max_score(scores))
    print(f'current generation is :{generation}')
    print("\nBest Route Found:")
    print(" -> ".join(chromosomes[best_index]) + f" -> {chromosomes[best_index][0]}")
    print(f"Total Distance: {1 / scores[best_index]:.2f}")
    print(f"time duration is : {(time.time()-start_time)*1000:.2f} ms")

    print("____________________________________")
    bt_start_time = time.time()
    bt_ans , bt_path = tsp_backtracking(DISTANCES)
    print(f"BACKTRACKING ANSWER IS : {bt_ans}")
    print(f"backtracking path is : {bt_path}")
    print(f"time duration is : {(time.time() - bt_start_time)*1000:.2f} ms")
##############################################################################PARAMETERS
data_path = select_file()
DISTANCES = read_graph_to_dict(data_path)
CITIES = [city for city in DISTANCES.keys()]
POP_SIZE = 100
MUTATION_RATE = 0.50
MAX_OPTIMUM_SCORE = 28
EXPECTED_ERR_PERC = 100
######################################################################################
if __name__ == "__main__":
    main()