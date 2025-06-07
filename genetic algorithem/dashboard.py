import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import time
from app import (fitness_population_selection, initial_chromosomes, 
                chromosome_measure, mutation, cross_over,greedy_selection,mutation_doublepoint,crossover_doubleposition 
                ,crossover_monoposition,crossover,DISTANCES,POP_SIZE, MUTATION_RATE, MAX_OPTIMUM_SCORE, EXPECTED_ERR_PERC)

# Set page config
st.set_page_config(page_title="TSP Genetic Algorithm Dashboard", layout="wide")

# Title
st.title("Traveling Salesman Problem - Genetic Algorithm Dashboard")

# Sidebar for parameters
st.sidebar.header("Hyperparameters")

# Population size slider
pop_size = st.sidebar.slider(
    "Population Size",
    min_value=2,
    max_value=100,
    value=POP_SIZE,
    step=2,
    help="Number of chromosomes in each generation"
)

# Mutation rate slider
mutation_rate = st.sidebar.slider(
    "Mutation Rate",
    min_value=0.0,
    max_value=1.0,
    value=MUTATION_RATE,
    step=0.01,
    help="Probability of mutation occurring"
)

# Maximum generations slider
max_generations = st.sidebar.slider(
    "Maximum Generations",
    min_value=0,
    max_value=100000,
    value=10000,
    step=100,
    help="Maximum number of generations to run"
)

# Expected error percentage slider
expected_err_perc = st.sidebar.slider(
    "Expected Error Percentage",
    min_value=1,
    max_value=100,
    value=EXPECTED_ERR_PERC,
    step=1,
    help="Expected error percentage for convergence"
)

# Maximum optimum score input
max_optimum_score = st.sidebar.number_input(
    "Maximum Optimum Score",
    min_value=1,
    max_value=1000,
    value=MAX_OPTIMUM_SCORE,
    help="Maximum optimum score for the problem"
)

# Add these selection boxes in the sidebar after your other parameters:
st.sidebar.header("Algorithm Variants")

# Selection method chooser
selection_method = st.sidebar.selectbox(
    "Selection Method",
    ["Roulette Wheel", "Greedy"],
    help="Choose the selection method for parent chromosomes"
)

# Mutation method chooser
mutation_method = st.sidebar.selectbox(
    "Mutation Method",
    ["Single Point", "Double Point"],
    help="Choose the mutation method"
)

# Crossover method chooser
crossover_method = st.sidebar.selectbox(
    "Crossover Method",
    ["Single Point", "Double Point", "Two Point"],
    help="Choose the crossover method"
)

# Add this after your parameter selection and before the run button
st.sidebar.markdown("---")
st.sidebar.subheader("Current Settings")
settings_info = f"""
- Selection: {selection_method}
- Mutation: {mutation_method}
- Crossover: {crossover_method}
"""
st.sidebar.markdown(settings_info)

def run_genetic_algorithm(pop_size, mutation_rate, max_generations, expected_err_perc, max_optimum_score,
                        selection_method, mutation_method, crossover_method):
    # Initialize metrics storage
    best_scores = []
    generations = []
    
    # Start timer
    start_time = time.time()
    
    # Initialize population
    chromosomes = initial_chromosomes()
    scores = chromosome_measure(chromosomes)
    generation = 0
    
    while generation < max_generations and max(scores) < expected_err_perc / 100 * (1 / max_optimum_score):
        # Selection based on chosen method
        if selection_method == "Roulette Wheel":
            chromosomes = fitness_population_selection(chromosomes, scores)
        else:  # Greedy
            chromosomes = greedy_selection(chromosomes, scores)
        
        # Crossover based on chosen method
        if crossover_method == "Single Point":
            new_population = []
            for _ in range(0, POP_SIZE, 2):
                parent1, parent2 = random.sample(chromosomes, 2)
                child1 = crossover_monoposition(parent1, parent2)
                child2 = crossover_monoposition(parent2, parent1)
                new_population.extend([child1, child2])
            chromosomes = new_population
        elif crossover_method == "Double Point":
            new_population = []
            for _ in range(0, POP_SIZE, 2):
                parent1, parent2 = random.sample(chromosomes, 2)
                child1 = crossover_doubleposition(parent1, parent2)
                child2 = crossover_doubleposition(parent2, parent1)
                new_population.extend([child1, child2])
            chromosomes = new_population
        else:  # Two Point
            new_population = []
            for _ in range(0, POP_SIZE, 2):
                parent1, parent2 = random.sample(chromosomes, 2)
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                new_population.extend([child1, child2])
            chromosomes = new_population
        
        # Mutation based on chosen method
        if mutation_method == "Single Point":
            chromosomes = mutation(chromosomes)
        else:  # Double Point
            chromosomes = mutation_doublepoint(chromosomes, DISTANCES)
        
        scores = chromosome_measure(chromosomes)
        
        # Store metrics (limit the score to 1000)
        current_best = min(1/max(scores), 1000)
        best_scores.append(current_best)
        generations.append(generation)
        
        generation += 1
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Get best route
    best_index = scores.index(max(scores))
    best_route = chromosomes[best_index]
    
    return generations, best_scores, execution_time, best_route

# Run button
if st.sidebar.button("Run Algorithm"):
    with st.spinner("Running genetic algorithm..."):
        # Run the algorithm with selected methods
        generations, best_scores, execution_time, best_route = run_genetic_algorithm(
            pop_size, 
            mutation_rate, 
            max_generations, 
            expected_err_perc, 
            max_optimum_score,
            selection_method,
            mutation_method,
            crossover_method
        )
        
        # Plot results
        st.subheader("Distance Over Generations")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot with improved styling
        ax.plot(generations, best_scores, 
                label='Best Distance', 
                color='blue',
                linewidth=2)
        
        # Set axis limits and labels
        ax.set_ylim(1, 80)  # Set y-axis range from 1 to 100
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Customize appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend
        ax.legend(fontsize=10)
        
        # Set title
        plt.title('Best Distance per Generation', pad=20, fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
        
        # Display final results in a nice format
        st.subheader("Final Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Execution Time", f"{execution_time:.2f} seconds")
            st.metric("Final Generation", len(generations))
            
        with col2:
            st.metric("Best Distance Found", f"{min(best_scores[-1], 1000):.2f}")
            st.metric("Number of Cities", len(best_route))
        
        # Display best route
        st.subheader("Best Route Found")
        st.write(" → ".join(best_route) + f" → {best_route[0]}")

# Footer
st.markdown("---")
st.markdown("""
    **Dashboard created by Mahdi Ahmadi**  
    GitHub: [mahdiahmadii](https://github.com/mahdiahmadii)  
    LinkedIn: [mahdiahmadii](https://linkedin.com/in/mahdiahmadii)
    Telegram: [mahdiahmadiE](https://t.me/mahdiahmadie)
""")