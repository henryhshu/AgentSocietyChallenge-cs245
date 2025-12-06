<div style="text-align: center; display: flex; align-items: center; justify-content: center; background-color: white; padding: 20px; border-radius: 30px;">
  <h1 style="color: black; margin: 0; font-size: 2em;">CS 245 AgentSociety Challenge: Track 1</h1>
</div>

# 🚀 AgentSociety Challenge Track 1 (User Modeling)
![License](https://img.shields.io/badge/license-MIT-green) &ensp;
[![arXiv](https://img.shields.io/badge/arXiv-2502.18754-b31b1b.svg)](https://arxiv.org/abs/2502.18754)

This repository provides the tools and framework needed to participate in a competition that focuses on building **LLM Agents** for **user behavior simulation** and **recommendation systems** based on open source datasets.

Participants are tasked with developing intelligent agents that interact with a simulated environment and perform specific tasks in two competition tracks:
1. **User Behavior Simulation Track**: Agents simulate user behavior, including generating reviews and ratings.
2. **Recommendation Track**: Agents generate recommendations based on provided contextual data.

Our group focused on track 1, which is simulating user behavior.

This repository includes the base repository for the challenge:
- The core library `websocietysimulator` for environment simulation.
- Scripts for dataset processing and analysis.
- Example usage for creating and evaluating agents.

It also includes our changes for improving the user simulation using the Reasoning, Planning, and Memory modules.

---

## Directory Structure

### 1. **`websocietysimulator/`**  
This is the core library containing all source code required for the competition.

- **`agents/`**: Contains base agent classes (`SimulationAgent`, `RecommendationAgent`) and their abstractions. Participants must extend these classes for their implementations.
- **`task/`**: Defines task structures for each track (`SimulationTask`, `RecommendationTask`).
- **`llm/`**: Contains base LLM client classes (`DeepseekLLM`, `OpenAILLM`).
- **`tools/`**: Includes utility tools:
  - `InteractionTool`: A utility for interacting with the Yelp dataset during simulations.
  - `EvaluationTool`: Provides comprehensive metrics for both recommendation (HR@1/3/5) and simulation tasks (RMSE, sentiment analysis).
- **`simulator.py`**: The main simulation framework, which handles task and groundtruth setting, evaluation and agent execution.

### 2. **`example/`**  
Contains usage examples of the `websocietysimulator` library. Includes sample agents and scripts to demonstrate how to load scenarios, set agents, and evaluate them.

### 3. **`data_process.py`**  
A script to process the raw Yelp dataset into the required format for use with the `websocietysimulator` library. This script ensures the dataset is cleaned and structured correctly for simulations.

---

## Quick Start

### 1. Install the Library

The repository is organized using [Python Poetry](https://python-poetry.org/). Follow these steps to install the library:

1. Clone the repository:
   ```bash
   git clone <this_repo>
   cd websocietysimulator
   ```

2. Install dependencies:
  - Option 1: Install dependencies using Poetry: (Recommended)
    ```bash
    poetry install  && \
    poetry shell
    ```
  - Option 2: Install dependencies using pip(COMING SOON):
    ```bash
    pip install websocietysimulator
    ```
  - Option 3: Install dependencies using conda:
    ```bash
    conda create -n websocietysimulator python=3.11 && \
    conda activate websocietysimulator && \
    pip install -r requirements.txt && \
    pip install .
    ```

3. Verify the installation:
   ```python
   import websocietysimulator
   ```

---

### 2. Data Preparation

1. Download the raw dataset from the Yelp[1], Amazon[2] or Goodreads[3].
2. Run the `data_process.py` script to process the dataset:
   ```bash
   python data_process.py --input <path_to_raw_dataset> --output <path_to_processed_dataset>
   ```
- Check out the [Data Preparation Guide](./tutorials/data_preparation.md) for more information.
- **NOTICE: You Need at least 16GB RAM to process the dataset.**

---

### 3. Organize Your Data

Ensure the dataset is organized in a directory structure similar to this:

```
<your_dataset_directory>/
├── item.json
├── review.json
├── user.json
```

You can name the dataset directory whatever you prefer (e.g., `dataset/`).

---

- Check out the [Tutorial](./tutorials/agent_development.md) for Agent Development.
- Baseline User Behavior Simulation Agent: [Baseline User Behavior Simulation Agent](./example/ModelingAgent_baseline.py).
- Baseline Recommendation Agent: [Baseline Recommendation Agent](./example/RecAgent_baseline.py).
---

### 4. Evaluationing your agent with training data

Run the simulation using the provided `Simulator` class:

```python
from websocietysimulator import Simulator
from my_agent import MySimulationAgent

# Initialize Simulator
simulator = Simulator(data_dir="path/to/your/dataset", device="auto", cache=False)
# The cache parameter controls whether to use cache for interaction tool.
# If you want to use cache, you can set cache=True. When using cache, the simulator will only load data into memory when it is needed, which saves a lot of memory.
# If you want to use normal interaction tool, you can set cache=False. Notice that, normal interaction tool will load all data into memory at the beginning, which needs a lot of memory (20GB+).

# Load scenarios
simulator.set_task_and_groundtruth(task_dir="path/to/task_directory", groundtruth_dir="path/to/groundtruth_directory")

# Set your custom agent
simulator.set_agent(MySimulationAgent)

# Set LLM client
simulator.set_llm(DeepseekLLM(api_key="Your API Key"))

# Run evaluation
# If you don't set the number of tasks, the simulator will run all tasks.
agent_outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=10)

# Evaluate the agent
evaluation_results = simulator.evaluate()
```
- If you want to use your own LLMClient, you can easily implement it by inheriting the `LLMBase` class. Refer to the [Tutorial](./tutorials/agent_development.md) for more information.

---

## Introduction to the `InteractionTool`

The `InteractionTool` is the core utility for interacting with the dataset. It provides an interface for querying user, item, and review data.

### Functions

- **Get User Information**:
  Retrieve user data by user ID or current scenario context.
  ```python
  user_info = interaction_tool.get_user(user_id="example_user_id")
  ```

- **Get Item Information**:
  Retrieve item data by item ID or current scenario context.
  ```python
  item_info = interaction_tool.get_item(item_id="example_item_id")
  ```

- **Get Reviews**:
  Fetch reviews related to a specific item or user, filtered by time.
  ```python
  reviews = interaction_tool.get_reviews(review_id="example_review_id")  # Fetch a specific review
  reviews = interaction_tool.get_reviews(item_id="example_item_id")  # Fetch all reviews for a specific item
  reviews = interaction_tool.get_reviews(user_id="example_user_id")  # Fetch all reviews for a specific user
  ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## References

[1] Yelp Dataset: https://www.yelp.com/dataset

[2] Amazon Dataset: https://amazon-reviews-2023.github.io/

[3] Goodreads Dataset: https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home
