# Interactive Dashboard

A comprehensive Streamlit dashboard for exploring space-time tradeoffs in computing systems.

## Features

### 1. Overview Page
- Visualizes Williams' theoretical bound: TIME[t] ⊆ SPACE[√(t log t)]
- Shows the fundamental space-time tradeoff curve
- Compares theoretical vs practical bounds

### 2. Theoretical Explorer
- Interactive parameter adjustment
- Real-time visualization of space requirements for given time bounds
- Constant factor analysis

### 3. Experimental Results
- **Maze Solver**: BFS vs memory-limited algorithms
- **Sorting**: In-memory vs checkpointed sorting
- **Streaming**: Sliding window performance
- Summary of all experimental findings

### 4. Real-World Systems
- **Databases**: Query optimization and join algorithms
- **LLMs**: Memory optimization techniques
- **Distributed Computing**: MapReduce and shuffle optimization

### 5. Tradeoff Calculator
- Input your system parameters
- Get recommendations for optimal configurations
- Compare different strategies

### 6. Interactive Demos
- Sorting visualizer
- Cache hierarchy simulator
- Live demonstrations of space-time tradeoffs

## Running the Dashboard

### Option 1: Using the launcher script
```bash
cd dashboard
python run_dashboard.py
```

### Option 2: Direct streamlit command
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

The dashboard will open in your default browser at http://localhost:8501

## Technology Stack

- **Streamlit**: Interactive web framework
- **Plotly**: Advanced interactive visualizations
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations

## Customization

The dashboard is fully customizable:
- Add new visualizations to `app.py`
- Modify color schemes in the CSS section
- Add new pages in the sidebar navigation
- Import real experimental data to replace simulated data

## Screenshots

The dashboard includes:
- Dark theme optimized for data visualization
- Responsive layout for different screen sizes
- Interactive controls for exploring parameters
- Real-time updates as you adjust settings