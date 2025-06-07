# Traveling Salesman Problem Solver using Genetic Algorithm

This project implements a genetic algorithm solution for the Traveling Salesman Problem (TSP) with an interactive dashboard for parameter tuning and visualization.

## Description

The Traveling Salesman Problem is a classic optimization problem where the goal is to find the shortest possible route that visits each city exactly once and returns to the starting city. This implementation uses genetic algorithms with the following features:

### Algorithm Features
- Multiple selection methods:
  - Roulette wheel selection
  - Greedy selection
- Various crossover operators:
  - Single-point crossover
  - Double-point crossover
  - Two-point crossover
- Different mutation strategies:
  - Single-point mutation
  - Double-point mutation
- Performance optimization techniques
- Visualization of results

### Dashboard Features
- Real-time parameter tuning
- Interactive visualization
- Algorithm variant selection
- Performance metrics tracking
- Best route display
- Execution time monitoring

## Installation

1. Extract the zip file to your desired location
2. Open Command Prompt or PowerShell as administrator
3. Navigate to the project directory
4. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Generating sample data
```bash
python data/data_generator.py #for auto generating
```
or if you want to do it manually
```bash 
python data/input_data.py #generating masually
```


### Running the Main Algorithm
```bash
python app.py
```

### Running the Dashboard
```bash
streamlit run dashboard.py
```

The dashboard allows you to:
- Adjust population size and mutation rate
- Set maximum generations and error percentage
- Choose between different selection methods
- Select mutation and crossover operators
- Visualize the optimization process in real-time
- Export results and visualizations

## Project Structure
```
project/
│
├── app.py # Main algorithm implementation
├── dashboard.py # Interactive dashboard
├── requirements.txt # Project dependencies
├── README.md # This file
├── unit_tests.ipynb # Unit tests and examples
│
├── data/ # Input data directory
│ ├── data_generator.py #ggenerating sample data
│ ├── input_data.py #adding data manually
│ ├── graph.csv # Sample graph data
│ └── graph1.csv # Additional sample data
```

## Dashboard Controls

### Hyperparameters
- Population Size (2-100)
- Mutation Rate (0-1)
- Maximum Generations
- Expected Error Percentage
- Maximum Optimum Score

### Algorithm Variants
- Selection Methods:
  - Roulette Wheel
  - Greedy
- Mutation Methods:
  - Single Point
  - Double Point
- Crossover Methods:
  - Single Point
  - Double Point
  - Two Point

### Visualization
- Distance over generations plot
- Best route found
- Execution time
- Performance metrics

## Testing

The project includes comprehensive unit tests in `unit_tests.ipynb`. This Jupyter notebook contains:
- Individual component tests
- Performance benchmarks
- Example usage scenarios
- Visualization examples

## Input Data

The `data/` directory contains sample input files in CSV format. You can:
- Use the provided sample data
- Create your own input files following the same format
- Modify existing files for different scenarios

## Author

**Mahdi Ahmadi**

- GitHub: [mahdiahmadii](https://github.com/mahdiahmadii)
- LinkedIn: [mahdiahmadii](https://linkedin.com/in/mahdiahmadii)
- Email: mahdi2002hmadi82@gmail.com
- Telegram: [mahdiahmadiE](https://t.me/mahdiahmadie)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

