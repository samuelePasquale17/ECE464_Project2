# SCOAP and Monte Carlo Simulation for Circuit Analysis

## Overview
This project implements **SCOAP (Sandia Controllability/Observability Analysis Program)** and **Monte Carlo (MC)** simulation methods to analyze circuit properties. The circuit is modeled as a **directed graph**, and these methods are applied to calculate controllability metrics and simulate circuit behavior, providing insights into the complexity of controlling circuit nodes.

---

## Key Features

### Circuit Modeling
- The circuit is modeled as a **directed graph**, where:
  - **Nodes**: Represent gates, inputs, and outputs.
  - **Edges**: Represent wires connecting the nodes.
- Node classification:
  - **Inputs**: Nodes with only outgoing edges.
  - **Outputs**: Nodes with only incoming edges.
  - **Intermediate Nodes**: Nodes with both incoming and outgoing edges.
- A **leveling function** assigns levels to nodes, starting with `lvl = 0` for inputs, enabling efficient traversal from inputs to outputs.

---

### SCOAP Method
- **Controllability Metrics**:
  - **c0**: Effort required to set a node to 0.
  - **c1**: Effort required to set a node to 1.
- **Process**:
  1. Nodes are traversed in increasing order of levels.
  2. Inputs are initialized with controllabilities `(c0, c1) = (1, 1)`.
  3. For each node:
     - Read controllability values from input nodes (connected via incoming edges).
     - Compute new `(c0, c1)` values based on gate type.
  4. Update the controllability dictionary with the computed values.
  5. Continue until all nodes have associated `(c0, c1)` values.

---

### Monte Carlo (MC) Simulation
- **Simulation Function**:
  - Extended to return both output values and internal node states when `ret_mc_sim=True` is enabled.
  - Tracks node states for the entire circuit during simulation.
- **Process**:
  1. Simulate the circuit for all possible input vectors (`2^n`, where `n` is the number of inputs).
  2. Return a dictionary mapping each node to its simulated boolean values.

---

### Comparison: SCOAP vs. MC
- **Analysis**:
  - Compare **SCOAP controllability percentages** with **MC simulation probabilities**:
    - SCOAP: `(c1 * 100) / (c1 + c0)`.
    - MC: `(p1 * 100) / (p1 + p0)` (from simulated values).
  - Plot the absolute difference between SCOAP and MC probabilities for each node.
- **Results**:
  - SCOAP metrics do not inherently represent probabilities.
  - Larger differences are observed in some nodes, showing that:
    - SCOAP indicates the **effort required to control a node**, not its probability of being 0 or 1.
    - Higher SCOAP controllability implies greater difficulty in controlling the node’s value.

---

## Tools and Implementation
- **Directed Graphs**:
  - Efficient modeling of circuits.
  - Simplifies traversal and node-level computations.
- **Python**:
  - Implementation of graph-based algorithms and simulations.
  - Data structures for controllability and simulation.

---

## Results
- **Execution Efficiency**:
  - Circuit modeling and leveling streamline node traversal.
  - MC simulation provides detailed insights into circuit behavior.
- **Key Insights**:
  - SCOAP controllability represents control difficulty, not probabilities.
  - Monte Carlo simulation complements SCOAP for probabilistic analysis.

---

## Conclusion
This project highlights the complementary roles of **SCOAP** and **MC simulation** in circuit analysis. While SCOAP focuses on the effort needed to control circuit nodes, MC provides probabilistic insights, enabling a deeper understanding of circuit behavior during testing and simulation.
