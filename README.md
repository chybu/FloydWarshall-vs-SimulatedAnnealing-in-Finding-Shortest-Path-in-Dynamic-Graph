# 🛰️ Floyd–Warshall vs Simulated Annealing for Shortest Path in Dynamic Graphs

A comparative study of the **Floyd–Warshall (FW)** algorithm and **Simulated Annealing (SA)** for solving **shortest path problems** in **traffic-based graphs**.  
Experiments are conducted on both **static** and **dynamic** networks using real-world data from **OpenStreetMap (OSM)**.

> Built with **Python 3.11**

---

## 📘 Overview

The **Floyd–Warshall (FW)** algorithm is a classical dynamic programming approach that computes all-pairs shortest paths in a weighted graph. It performs efficiently on smaller, static graphs but becomes computationally expensive for large or frequently changing networks.

In contrast, **Simulated Annealing (SA)** is a **metaheuristic optimization technique** inspired by the physical annealing process. It uses **randomized exploration, adaptive temperature control,** and **mutation** to find near-optimal solutions in complex or dynamic environments. While SA is commonly applied to the **Traveling Salesman Problem (TSP)**, this project explores its potential in solving **shortest path problems (SPP)** as well.

To evaluate both algorithms, FW is used as a deterministic baseline while SA acts as a heuristic alternative. Both methods operate on **directed adjacency matrices**.

---

## 🧩 Testing Scenarios

1. **Graph generation**  
   - Graphs are generated from the **OpenStreetMap API** using `osmnx`.  
   - Nodes represent **road intersections**, and edges represent **roads** between them.  
   - Edge weights are computed using three factors:  
     - **Length** of the road  
     - **Maximum allowed speed**  
     - **Number of lanes**

2. **Graph density**  
   - Real-world road networks are typically sparse, with a **density < 0.08**.  
   - To achieve more diverse testing conditions, additional **synthetic graphs** are generated with predefined densities.

3. **Static graphs**  
   - Edge weights remain constant throughout the test.

4. **Dynamic graphs**  
   - Edge weights **change periodically** to simulate traffic variations.  
   - Since FW and SA converge at different rates, their underlying graphs evolve differently.  
   - After both algorithms converge, **Dijkstra’s algorithm** is applied to the final state of each graph to compute the **ground-truth shortest path**.

---

## ⚙️ Evaluation Metrics

- **Runtime** – total execution time.  
- **Memory usage** – total RAM consumed during execution.  
- **Error from optimal** – difference from Dijkstra’s ground-truth path.  
- **Improvement Rate (SA only)** – how much SA improves its initial solution.  
- **Fast Convergence Score (SA only)** – how quickly SA finds a better solution (range: 0–1, lower is faster).

---

## 📂 Repository Structure

| File | Description |
|------|--------------|
| `graph_generator.py` | Generates adjacency matrices using OpenStreetMap data. |
| `map_generator.py` | Visualizes FW and SA paths compared with Dijkstra’s result. |
| `random_dynamic.py` | Compares FW and SA on **dynamic** random graphs. |
| `random_static.py` | Compares FW and SA on **static** random graphs. |
| `real_dynamic.py` | Compares FW and SA on **dynamic** real-world traffic graphs. |
| `real_static.py` | Compares FW and SA on **static** real-world traffic graphs. |

---

## 📄 Reference

This repository contains the source code for the paper:  
📘 [**“Floyd–Warshall vs. Simulated Annealing in Finding Shortest Path in Dynamic Graphs”**](https://ieeexplore.ieee.org/abstract/document/11121623)
