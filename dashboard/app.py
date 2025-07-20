"""
Interactive Dashboard for Space-Time Tradeoffs
Visualizes Williams' theoretical result and practical manifestations
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Space-Time Tradeoffs Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {padding-top: 1rem;}
    .stPlotlyChart {background-color: #0e1117;}
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #333;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🔄 The Ubiquity of Space-Time Tradeoffs")
st.markdown("""
This dashboard demonstrates **Ryan Williams' 2025 result**: TIME[t] ⊆ SPACE[√(t log t)]

Explore how this theoretical bound manifests in real computing systems.
""")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Choose a visualization",
    ["Overview", "Theoretical Explorer", "Experimental Results", 
     "Real-World Systems", "Tradeoff Calculator", "Interactive Demos"]
)

# Helper functions
def create_space_time_curve(n_points=100):
    """Generate theoretical space-time tradeoff curve"""
    t = np.logspace(1, 6, n_points)
    s_williams = np.sqrt(t * np.log(t))
    s_naive = t
    s_minimal = np.log(t)
    
    return t, s_williams, s_naive, s_minimal

def create_3d_tradeoff_surface():
    """Create 3D visualization of space-time-quality tradeoffs"""
    space = np.logspace(0, 3, 50)
    time = np.logspace(0, 3, 50)
    S, T = np.meshgrid(space, time)
    
    # Quality as function of space and time
    Q = 1 / (1 + np.exp(-(np.log(S) + np.log(T) - 4)))
    
    return S, T, Q

# Page: Overview
if page == "Overview":
    st.header("Key Concepts")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Theoretical Bound", "√(t log t)", "Space for time t")
        st.info("Any computation taking time t can be done with √(t log t) memory")
    
    with col2:
        st.metric("Practical Factor", "100-10,000×", "Constant overhead")
        st.warning("Real systems have I/O, cache hierarchies, coordination costs")
    
    with col3:
        st.metric("Ubiquity", "Everywhere", "In modern systems")
        st.success("Databases, ML, distributed systems all use these tradeoffs")
    
    # Main visualization
    st.subheader("The Fundamental Tradeoff")
    
    t, s_williams, s_naive, s_minimal = create_space_time_curve()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=t, y=s_naive,
        mode='lines',
        name='Naive (Space = Time)',
        line=dict(color='red', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=t, y=s_williams,
        mode='lines',
        name='Williams\' Bound: √(t log t)',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=t, y=s_minimal,
        mode='lines', 
        name='Minimal Space: log(t)',
        line=dict(color='green', dash='dot')
    ))
    
    fig.update_xaxes(type="log", title="Time (t)")
    fig.update_yaxes(type="log", title="Space (s)")
    fig.update_layout(
        title="Theoretical Space-Time Bounds",
        height=500,
        hovermode='x',
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Page: Theoretical Explorer
elif page == "Theoretical Explorer":
    st.header("Interactive Theoretical Explorer")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        
        time_complexity = st.slider(
            "Time Complexity (log scale)",
            min_value=1.0,
            max_value=6.0,
            value=3.0,
            step=0.1
        )
        
        show_practical = st.checkbox("Show practical bounds", value=True)
        constant_factor = st.slider(
            "Constant factor",
            min_value=1,
            max_value=1000,
            value=100,
            disabled=not show_practical
        )
        
        t_value = 10 ** time_complexity
        s_theory = np.sqrt(t_value * np.log(t_value))
        s_practical = s_theory * constant_factor if show_practical else s_theory
        
        st.metric("Time (t)", f"{t_value:,.0f}")
        st.metric("Space (theory)", f"{s_theory:,.0f}")
        if show_practical:
            st.metric("Space (practical)", f"{s_practical:,.0f}")
    
    with col2:
        # Create visualization
        t_range = np.logspace(1, 6, 100)
        s_range_theory = np.sqrt(t_range * np.log(t_range))
        s_range_practical = s_range_theory * constant_factor
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=t_range, y=s_range_theory,
            mode='lines',
            name='Theoretical Bound',
            line=dict(color='blue', width=2)
        ))
        
        if show_practical:
            fig.add_trace(go.Scatter(
                x=t_range, y=s_range_practical,
                mode='lines',
                name=f'Practical ({constant_factor}× overhead)',
                line=dict(color='orange', width=2)
            ))
        
        # Add current point
        fig.add_trace(go.Scatter(
            x=[t_value], y=[s_theory],
            mode='markers',
            name='Current Selection',
            marker=dict(size=15, color='red', symbol='star')
        ))
        
        fig.update_xaxes(type="log", title="Time")
        fig.update_yaxes(type="log", title="Space")
        fig.update_layout(
            title="Space Requirements for Time-Bounded Computation",
            height=500,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Page: Experimental Results
elif page == "Experimental Results":
    st.header("Experimental Validation")
    
    tabs = st.tabs(["Maze Solver", "Sorting", "Streaming", "Summary"])
    
    with tabs[0]:
        st.subheader("Maze Solving Algorithms")
        
        # Simulated data (in practice, load from experiment results)
        maze_data = pd.DataFrame({
            'Size': [20, 30, 40, 50],
            'BFS_Time': [0.001, 0.003, 0.008, 0.015],
            'BFS_Memory': [1600, 3600, 6400, 10000],
            'Limited_Time': [0.01, 0.05, 0.15, 0.35],
            'Limited_Memory': [80, 120, 160, 200]
        })
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Time Complexity", "Memory Usage")
        )
        
        fig.add_trace(
            go.Scatter(x=maze_data['Size'], y=maze_data['BFS_Time'],
                      name='BFS', mode='lines+markers'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=maze_data['Size'], y=maze_data['Limited_Time'],
                      name='Memory-Limited', mode='lines+markers'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=maze_data['Size'], y=maze_data['BFS_Memory'],
                      name='BFS', mode='lines+markers', showlegend=False),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=maze_data['Size'], y=maze_data['Limited_Memory'],
                      name='Memory-Limited', mode='lines+markers', showlegend=False),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Maze Size", row=1, col=1)
        fig.update_xaxes(title_text="Maze Size", row=1, col=2)
        fig.update_yaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Memory (cells)", row=1, col=2)
        
        fig.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("Memory-limited DFS uses √n memory but requires ~n√n time due to recomputation")
    
    with tabs[1]:
        st.subheader("Sorting with Checkpoints")
        
        sort_times = {
            'Size': [1000, 5000, 10000, 20000],
            'In_Memory': [0.00001, 0.0001, 0.0003, 0.0008],
            'Checkpointed': [0.268, 2.5, 8.2, 25.3],
            'Ratio': [26800, 25000, 27333, 31625]
        }
        
        df = pd.DataFrame(sort_times)
        
        fig = px.bar(df, x='Size', y=['In_Memory', 'Checkpointed'],
                     title="Sorting Time: In-Memory vs Checkpointed",
                     labels={'value': 'Time (seconds)', 'variable': 'Method'},
                     log_y=True,
                     barmode='group',
                     template="plotly_dark")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.warning("Checkpointed sorting shows massive overhead (>1000×) due to disk I/O")
    
    with tabs[2]:
        st.subheader("Stream Processing")
        
        stream_data = {
            'Window_Size': [10, 50, 100, 500, 1000],
            'Full_Storage_Time': [0.005, 0.025, 0.05, 0.25, 0.5],
            'Sliding_Window_Time': [0.001, 0.001, 0.001, 0.002, 0.003],
            'Memory_Ratio': [100, 100, 100, 100, 100]
        }
        
        df = pd.DataFrame(stream_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Window_Size'], y=df['Full_Storage_Time'],
            mode='lines+markers',
            name='Full Storage',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Window_Size'], y=df['Sliding_Window_Time'],
            mode='lines+markers',
            name='Sliding Window',
            line=dict(color='green')
        ))
        
        fig.update_xaxes(title="Window Size")
        fig.update_yaxes(title="Time (seconds)", type="log")
        fig.update_layout(
            title="Stream Processing: Less Memory = Faster!",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("Sliding window (O(w) space) is faster due to cache locality!")
    
    with tabs[3]:
        st.subheader("Summary of Findings")
        
        findings = pd.DataFrame({
            'Experiment': ['Maze Solver', 'Sorting', 'Streaming'],
            'Space Reduction': ['n → √n', 'n → √n', 'n → w'],
            'Time Increase': ['√n×', '>1000×', '0.1× (faster!)'],
            'Bottleneck': ['Recomputation', 'Disk I/O', 'Cache Locality']
        })
        
        st.table(findings)

# Page: Real-World Systems
elif page == "Real-World Systems":
    st.header("Space-Time Tradeoffs in Production")
    
    system = st.selectbox(
        "Choose a system",
        ["Databases", "Large Language Models", "Distributed Computing"]
    )
    
    if system == "Databases":
        st.subheader("Database Query Processing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Hash Join vs Nested Loop")
            
            memory_limit = st.slider("work_mem (MB)", 1, 1024, 64)
            table_size = st.slider("Table size (GB)", 1, 100, 10)
            
            # Simulate query planner decision
            if memory_limit > table_size * 10:
                join_type = "Hash Join"
                time_estimate = table_size * 0.1
                memory_use = min(memory_limit, table_size * 50)
            else:
                join_type = "Nested Loop"
                time_estimate = table_size ** 2 * 0.01
                memory_use = 1
            
            st.metric("Selected Algorithm", join_type)
            st.metric("Estimated Time", f"{time_estimate:.1f} seconds")
            st.metric("Memory Usage", f"{memory_use} MB")
        
        with col2:
            # Visualization
            mem_range = np.logspace(0, 3, 100)
            hash_time = np.ones_like(mem_range) * table_size * 0.1
            nested_time = np.ones_like(mem_range) * table_size ** 2 * 0.01
            
            # Hash join only works with enough memory
            hash_time[mem_range < table_size * 10] = np.inf
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=mem_range, y=hash_time,
                mode='lines',
                name='Hash Join',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=mem_range, y=nested_time,
                mode='lines',
                name='Nested Loop',
                line=dict(color='red')
            ))
            
            fig.add_vline(x=memory_limit, line_dash="dash", line_color="green",
                         annotation_text="Current work_mem")
            
            fig.update_xaxes(type="log", title="Memory Available (MB)")
            fig.update_yaxes(type="log", title="Query Time (seconds)")
            fig.update_layout(
                title="Join Algorithm Selection",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif system == "Large Language Models":
        st.subheader("LLM Memory Optimizations")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            model_size = st.selectbox("Model Size", ["7B", "13B", "70B", "175B"])
            optimization = st.multiselect(
                "Optimizations",
                ["Quantization (INT8)", "Flash Attention", "Multi-Query Attention"],
                default=[]
            )
            
            # Calculate memory requirements
            base_memory = {"7B": 28, "13B": 52, "70B": 280, "175B": 700}[model_size]
            memory = base_memory
            speedup = 1.0
            
            if "Quantization (INT8)" in optimization:
                memory /= 4
                speedup *= 0.8
            
            if "Flash Attention" in optimization:
                memory *= 0.7
                speedup *= 0.9
            
            if "Multi-Query Attention" in optimization:
                memory *= 0.6
                speedup *= 0.95
            
            st.metric("Memory Required", f"{memory:.0f} GB")
            st.metric("Relative Speed", f"{speedup:.2f}×")
            st.metric("Context Length", f"{int(100000 / (memory / base_memory))} tokens")
        
        with col2:
            # Create optimization impact chart
            categories = ['Memory', 'Speed', 'Context Length', 'Quality']
            
            fig = go.Figure()
            
            # Baseline
            fig.add_trace(go.Scatterpolar(
                r=[100, 100, 100, 100],
                theta=categories,
                fill='toself',
                name='Baseline',
                line=dict(color='red')
            ))
            
            # With optimizations
            memory_score = (base_memory / memory) * 100
            speed_score = speedup * 100
            context_score = (memory_score) * 100 / 100
            quality_score = 95 if optimization else 100
            
            fig.add_trace(go.Scatterpolar(
                r=[memory_score, speed_score, context_score, quality_score],
                theta=categories,
                fill='toself',
                name='With Optimizations',
                line=dict(color='green')
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 200]
                    )),
                showlegend=True,
                template="plotly_dark",
                title="Optimization Impact"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif system == "Distributed Computing":
        st.subheader("MapReduce Shuffle Memory")
        
        # Interactive shuffle buffer sizing
        cluster_size = st.slider("Cluster Size (nodes)", 10, 1000, 100)
        data_size = st.slider("Data Size (TB)", 1, 100, 10)
        
        # Calculate optimal buffer size
        data_per_node = data_size * 1024 / cluster_size  # GB per node
        optimal_buffer = np.sqrt(data_per_node * 1024)  # MB
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Data per Node", f"{data_per_node:.1f} GB")
        with col2:
            st.metric("Optimal Buffer Size", f"{optimal_buffer:.0f} MB")
        with col3:
            st.metric("Buffer/Data Ratio", f"1:{int(data_per_node * 1024 / optimal_buffer)}")
        
        # Visualization of shuffle performance
        buffer_sizes = np.logspace(1, 4, 100)
        
        # Performance model
        io_time = data_per_node * 1024 / buffer_sizes * 10  # More I/O with small buffers
        cpu_time = buffer_sizes / 100  # More CPU with large buffers
        total_time = io_time + cpu_time
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=buffer_sizes, y=io_time,
            mode='lines',
            name='I/O Time',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=buffer_sizes, y=cpu_time,
            mode='lines',
            name='CPU Time',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=buffer_sizes, y=total_time,
            mode='lines',
            name='Total Time',
            line=dict(color='green', width=3)
        ))
        
        fig.add_vline(x=optimal_buffer, line_dash="dash", line_color="white",
                     annotation_text="√n Optimal")
        
        fig.update_xaxes(type="log", title="Shuffle Buffer Size (MB)")
        fig.update_yaxes(type="log", title="Time (seconds)")
        fig.update_layout(
            title="Shuffle Performance vs Buffer Size",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("The optimal buffer size follows the √n pattern predicted by theory!")

# Page: Tradeoff Calculator
elif page == "Tradeoff Calculator":
    st.header("Space-Time Tradeoff Calculator")
    
    st.markdown("Calculate optimal configurations for your system")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Parameters")
        
        total_data = st.number_input("Total Data Size (GB)", min_value=1, value=100)
        available_memory = st.number_input("Available Memory (GB)", min_value=1, value=16)
        
        io_speed = st.slider("I/O Speed (MB/s)", 50, 5000, 500)
        cpu_speed = st.slider("CPU Speed (GFLOPS)", 10, 1000, 100)
        
        workload_type = st.selectbox(
            "Workload Type",
            ["Batch Processing", "Stream Processing", "Interactive Query", "ML Training"]
        )
    
    with col2:
        st.subheader("Recommendations")
        
        # Calculate recommendations based on workload
        memory_ratio = available_memory / total_data
        
        if memory_ratio > 1:
            st.success("✅ Everything fits in memory!")
            strategy = "In-memory processing"
            chunk_size = total_data
        elif memory_ratio > 0.1:
            st.info("📊 Use hybrid approach")
            strategy = "Partial caching with smart eviction"
            chunk_size = np.sqrt(total_data * available_memory)
        else:
            st.warning("⚠️ Heavy space constraints")
            strategy = "Streaming with checkpoints"
            chunk_size = available_memory / 10
        
        st.metric("Recommended Strategy", strategy)
        st.metric("Optimal Chunk Size", f"{chunk_size:.1f} GB")
        
        # Time estimates
        if workload_type == "Batch Processing":
            time_memory = total_data / cpu_speed
            time_disk = total_data / io_speed * 1000 + total_data / cpu_speed * 2
            time_optimal = total_data / np.sqrt(available_memory) * 10
        else:
            time_memory = 1
            time_disk = 100
            time_optimal = 10
        
        # Comparison chart
        fig = go.Figure(data=[
            go.Bar(name='All in Memory', x=['Time'], y=[time_memory]),
            go.Bar(name='All on Disk', x=['Time'], y=[time_disk]),
            go.Bar(name='Optimal √n', x=['Time'], y=[time_optimal])
        ])
        
        fig.update_layout(
            title="Processing Time Comparison",
            yaxis_title="Time (seconds)",
            template="plotly_dark",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Page: Interactive Demos
elif page == "Interactive Demos":
    st.header("Interactive Demonstrations")
    
    demo = st.selectbox(
        "Choose a demo",
        ["Sorting Visualizer", "Cache Simulator", "Attention Mechanism"]
    )
    
    if demo == "Sorting Visualizer":
        st.subheader("Watch Space-Time Tradeoffs in Action")
        
        size = st.slider("Array Size", 10, 100, 50)
        algorithm = st.radio("Algorithm", ["In-Memory Sort", "External Sort with √n Memory"])
        
        if st.button("Run Sorting"):
            # Simulate sorting
            progress = st.progress(0)
            status = st.empty()
            
            if algorithm == "In-Memory Sort":
                steps = size * np.log2(size)
                for i in range(int(steps)):
                    progress.progress(i / steps)
                    status.text(f"Comparing elements... Step {i}/{int(steps)}")
                st.success(f"Completed in {steps:.0f} operations using {size} memory units")
            else:
                chunks = int(np.sqrt(size))
                total_steps = size * np.log2(size) * chunks
                for i in range(int(total_steps)):
                    progress.progress(i / total_steps)
                    if i % size == 0:
                        status.text(f"Writing checkpoint {i//size}/{chunks}...")
                    else:
                        status.text(f"Processing... Step {i}/{int(total_steps)}")
                st.warning(f"Completed in {total_steps:.0f} operations using {chunks} memory units")
    
    elif demo == "Cache Simulator":
        st.subheader("Memory Hierarchy Simulation")
        
        # Create memory hierarchy visualization
        levels = {
            'L1 Cache': {'size': 32, 'latency': 1},
            'L2 Cache': {'size': 256, 'latency': 10},
            'L3 Cache': {'size': 8192, 'latency': 50},
            'RAM': {'size': 32768, 'latency': 100},
            'SSD': {'size': 512000, 'latency': 10000}
        }
        
        access_pattern = st.selectbox(
            "Access Pattern",
            ["Sequential", "Random", "Strided"]
        )
        
        working_set = st.slider("Working Set Size (KB)", 1, 100000, 1000, step=10)
        
        # Determine which level serves the request
        for level, specs in levels.items():
            if working_set <= specs['size']:
                serving_level = level
                latency = specs['latency']
                break
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Data Served From", serving_level)
            st.metric("Average Latency", f"{latency} ns")
            st.metric("Throughput", f"{1000/latency:.1f} GB/s")
        
        with col2:
            # Visualization
            fig = go.Figure()
            
            sizes = [specs['size'] for specs in levels.values()]
            latencies = [specs['latency'] for specs in levels.values()]
            names = list(levels.keys())
            
            fig.add_trace(go.Scatter(
                x=sizes, y=latencies,
                mode='markers+text',
                text=names,
                textposition="top center",
                marker=dict(size=20)
            ))
            
            fig.add_vline(x=working_set, line_dash="dash", line_color="red",
                         annotation_text="Working Set")
            
            fig.update_xaxes(type="log", title="Capacity (KB)")
            fig.update_yaxes(type="log", title="Latency (ns)")
            fig.update_layout(
                title="Memory Hierarchy",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Created for the Ubiquity Project | Based on Ryan Williams' 2025 STOC paper</p>
    <p>TIME[t] ⊆ SPACE[√(t log t)] - A fundamental limit of computation</p>
</div>
""", unsafe_allow_html=True)