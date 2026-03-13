[//]: # (# TRL - Transformer Reinforcement Learning)

[//]: # ()
[//]: # (![Image]&#40;https://github.com/user-attachments/assets/3812473c-7c5b-4f59-b849-b9b77592d3c4&#41;)

[//]: # ()
[//]: # (# 🌌 Custom TRL: Elegance and Power in Reinforcement Learning)

[//]: # ()
[//]: # (Welcome to a project where fundamental mathematics meets cutting-edge artificial intelligence. This library is not just a custom analogue of the popular Hugging Face solution; it is a deeply reimagined, elegant, and crystal-clear architecture for fine-tuning Large Language Models &#40;LLMs&#41; using the **PPO &#40;Proximal Policy Optimization&#41;** algorithm.)

[//]: # ()
[//]: # (We believe that the process of creating AI should be as beautiful as the result of its work. This repository was forged for researchers and engineers who refuse to settle for "black boxes" and strive for absolute control over every tensor, every optimization step, and every nuance of the neural network alignment process &#40;RLHF&#41;.)

[//]: # ()
[//]: # (Unlock the true potential of your models, guiding their generation with surgical precision and observing the process through the lens of flawless analytics.)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## ✨ The Art of Technology: Key Features)

[//]: # ()
[//]: # (* **Architectural Purity and Freedom:** A completely open and intuitive PPO pipeline. You gain unlimited power over calculating KL divergence, estimating advantages, and shaping the surrogate loss function. No hidden abstractions—just the pure logic of the algorithm.)

[//]: # (* **A Symphony of Metrics with Weights & Biases &#40;W&B&#41;:** Watch your model train in real-time. Our native and deep integration with W&B transforms dry numbers into mesmerizing convergence graphs and interactive `game_log` tables, where every generated token and earned reward unfolds right before your eyes.)

[//]: # (* **Generation Without Borders:** Built-in and perfected sampling mechanisms &#40;Top-K and Nucleus&#41; allow your models to balance on the fine line between absolute creativity and strict logical coherence.)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## 🚀 Magic in One Click: Launch in Google Colab)

[//]: # ()
[//]: # (We understand that time is a researcher's most valuable resource. You don't need to spend hours setting up a local environment, hunting for available GPUs, and resolving dependencies. )

[//]: # ()
[//]: # (We have prepared an interactive Jupyter Notebook for you, where you can touch the training process directly in your browser. Launch a full RLHF cycle, experiment with hyperparameters, and watch your model evolve in real-time.)

[//]: # ()
[//]: # ([![Open In Colab]&#40;https://colab.research.google.com/assets/colab-badge.svg&#41;]&#40;https://colab.research.google.com/drive/1VvLGbyk4pvS1M1lIumQ3Awr-OjPfUUmj?usp=sharing&#41;)

[//]: # ()
[//]: # (> *Simply click the badge above, make a copy of the notebook to your Google Drive, and let the magic of training begin.*)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## 📦 Elegant Installation)

[//]: # ()
[//]: # (If you prefer to work in your own laboratory, deploying the project takes only a few moments. Clone the repository and install the dependencies:)

[//]: # ()
[//]: # (```bash)

[//]: # (!pip install "git+https://github.com/TimaGitHub/trl.git")

[//]: # (```)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## 📊 Visualizing Triumph: W&B Integration)

[//]: # ()
[//]: # (Reinforcement learning for neural networks is a complex dance between policy and value. Our library makes this dance visible. )

[//]: # ()
[//]: # (By activating the `use_wandb=True` flag, your dashboard transforms into a true mission control center:)

[//]: # (1. **The Aesthetics of Graphs:** Watch as the average reward &#40;`reward_mean`&#41; smoothly rises and how the system elegantly keeps KL divergence within set boundaries, preventing the model from losing its grip on reality.)

[//]: # (2. **Interactive Chronicles &#40;Game Log&#41;:** Each epoch leaves a mark in the form of a beautiful table. Read the input prompts, evaluate the generated texts, and analyze the rewards. These are not just logs—they are the coming-of-age story of your AI.)

[//]: # ()
[//]: # (Create, experiment, and inspire!)

[//]: # (# TRL - Transformer Reinforcement Learning)

[//]: # ()
[//]: # (![Image]&#40;https://github.com/user-attachments/assets/3812473c-7c5b-4f59-b849-b9b77592d3c4&#41;)

[//]: # ()
[//]: # ()
[//]: # (# 🧠 Custom LLM Alignment & Optimization )

[//]: # ()
[//]: # ([![Open In Colab]&#40;https://colab.research.google.com/assets/colab-badge.svg&#41;]&#40;https://colab.research.google.com/drive/1VvLGbyk4pvS1M1lIumQ3Awr-OjPfUUmj?usp=sharing&#41;)

[//]: # ()
[//]: # (Welcome to my educational repository dedicated to the inner workings of Large Language Model &#40;LLM&#41; alignment and fine-tuning. )

[//]: # ()
[//]: # (This project is built from scratch to step away from popular "black-box" libraries &#40;like Hugging Face `trl` or `peft`&#41; and gain a deep, fundamental understanding of the mathematics, tensor operations, and memory handling required to train modern AI. )

[//]: # ()
[//]: # (It is designed as an open laboratory for implementing, testing, and debugging state-of-the-art algorithms in PyTorch.)

[//]: # ()
[//]: # (## 🚀 Roadmap & Features)

[//]: # ()
[//]: # (This repository is actively growing. Here is the current implementation status of the core algorithms:)

[//]: # ()
[//]: # (* **✅ PPO &#40;Proximal Policy Optimization&#41;**)

[//]: # (  * Fully custom `PPOTrainer` pipeline.)

[//]: # (  * Generalized Advantage Estimation &#40;GAE&#41; and Monte Carlo returns.)

[//]: # (  * Adaptive KL-divergence penalty controllers to prevent reward hacking.)

[//]: # (  * Clipped surrogate objective and Value Function clipping.)

[//]: # (* **✅ DPO &#40;Direct Preference Optimization&#41;**)

[//]: # (  * Custom `DPOTrainer` implementing the exact paper formulation.)

[//]: # (  * Precise token-level math using Shift & Gather techniques for auto-regressive models.)

[//]: # (  * Proper loss masking &#40;ignoring prompt and padding tokens in the loss calculation&#41;.)

[//]: # (* **🚧 GRPO &#40;Group Relative Policy Optimization&#41;** — *[Work in Progress]*)

[//]: # (* **🚧 LoRA &#40;Low-Rank Adaptation&#41;** — *[Planned: Implementing custom forward passes with low-rank matrices]*)

[//]: # (* **🚧 Custom KV-Cache** — *[Planned: Deep dive into inference optimization]*)

[//]: # ()
[//]: # (## 🔬 Why this project?)

[//]: # ()
[//]: # (When using standard wrappers, complex operations like padding handling, log-probability extraction, and advantage calculation are hidden behind simple APIs. This project brings those complexities to the surface:)

[//]: # (- **Transparent Math:** Every equation from the original papers &#40;like DPO's implicit reward or PPO's surrogate loss&#41; is visibly translated into PyTorch tensor operations.)

[//]: # (- **Full Control:** Absolute freedom over data collation, masking strategies, and sampling mechanisms &#40;Top-K, Nucleus&#41;.)

[//]: # ()
[//]: # (## 📊 Weights & Biases Integration)

[//]: # ()
[//]: # (Reinforcement learning is notoriously difficult to debug. This repository natively integrates with W&B &#40;`use_wandb=True`&#41; to make the training process fully observable:)

[//]: # (- Track core metrics: KL divergence, Value Loss, Entropy, and Implicit Rewards &#40;Margin & Accuracy for DPO&#41;.)

[//]: # (- **Interactive Game Logs:** Custom tables log the `query`, `response`, and `reward` for every epoch, allowing you to qualitatively evaluate the model's behavior as it learns.)

[//]: # ()
[//]: # (## 💻 Quick Start)

[//]: # ()
[//]: # (You can explore the training process directly in your browser using the provided Google Colab notebook. It sets up the environment and launches a full RLHF/DPO cycle.)

[//]: # ()
[//]: # (> *Click the Colab badge at the top of this file to make a copy to your Google Drive and start experimenting.*)

[//]: # ()
[//]: # (### Installation)

[//]: # ()
[//]: # (To run this locally or in your own environment:)

[//]: # ()
[//]: # (```bash)

[//]: # (pip install "git+[https://github.com/TimaGitHub/trl.git]&#40;https://github.com/TimaGitHub/trl.git&#41;")
# TRL - Transformer Reinforcement Learning

![Image](https://github.com/user-attachments/assets/3812473c-7c5b-4f59-b849-b9b77592d3c4)


# 🧠 Custom LLM Alignment & Optimization 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VvLGbyk4pvS1M1lIumQ3Awr-OjPfUUmj?usp=sharing)

Welcome to my educational repository dedicated to the inner workings of Large Language Model (LLM) alignment and fine-tuning. 

This project is built from scratch to step away from popular "black-box" libraries (like Hugging Face `trl` or `peft`) and gain a deep, fundamental understanding of the mathematics, tensor operations, and memory handling required to train modern AI. 

It is designed as an open laboratory for implementing, testing, and debugging state-of-the-art algorithms in PyTorch.

## 🚀 Roadmap & Features

This repository is actively growing. Here is the current implementation status of the core algorithms:

* **✅ PPO (Proximal Policy Optimization)**
  * Fully custom `PPOTrainer` pipeline.
  * Generalized Advantage Estimation (GAE) and Monte Carlo returns.
  * Adaptive KL-divergence penalty controllers to prevent reward hacking.
  * Clipped surrogate objective and Value Function clipping.
* **✅ DPO (Direct Preference Optimization)**
  * Custom `DPOTrainer` implementing the exact paper formulation.
  * Precise token-level math using Shift & Gather techniques for auto-regressive models.
  * Proper loss masking (ignoring prompt and padding tokens in the loss calculation).
* **✅ GRPO (Group Relative Policy Optimization)**
  * Custom `GRPOTrainer` for efficient alignment without a separate value model.
  * Group-based advantage estimation using Z-score normalization.
  * Memory-efficient generation and optimization.
* **🚧 LoRA (Low-Rank Adaptation)** — *[Planned: Implementing custom forward passes with low-rank matrices]*

## 🔬 Why this project?

When using standard wrappers, complex operations like padding handling, log-probability extraction, and advantage calculation are hidden behind simple APIs. This project brings those complexities to the surface:
- **Transparent Math:** Every equation from the original papers (like DPO's implicit reward or PPO's surrogate loss) is visibly translated into PyTorch tensor operations.
- **Full Control:** Absolute freedom over data collation, masking strategies, and sampling mechanisms (Top-K, Nucleus).

## 📊 Weights & Biases Integration

Reinforcement learning is notoriously difficult to debug. This repository natively integrates with W&B (`use_wandb=True`) to make the training process fully observable:
- Track core metrics: KL divergence, Value Loss, Entropy, and Implicit Rewards (Margin & Accuracy for DPO).
- **Interactive Game Logs:** Custom tables log the `query`, `response`, and `reward` for every epoch, allowing you to qualitatively evaluate the model's behavior as it learns.

## 💻 Quick Start

You can explore the training process directly in your browser using the provided Google Colab notebook. It sets up the environment and launches a full RLHF/DPO cycle.

> *Click the Colab badge at the top of this file to make a copy to your Google Drive and start experimenting.*

### Installation

To run this locally or in your own environment:

```bash
pip install "git+[https://github.com/TimaGitHub/trl.git](https://github.com/TimaGitHub/trl.git)"