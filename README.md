

# Tic-Tac-Toe Reinforcement Learning

## Project Overview

This project implements a Tic-Tac-Toe game using Reinforcement Learning.

The main objective is to train intelligent agents that learn how to play Tic-Tac-Toe through repeated interactions with the game environment.

The project implements and compares the following Reinforcement Learning algorithms:

- Q-Learning
- SARSA
- Expected SARSA
- 3-step SARSA
- 3-step Off-Policy Learning

The performance of the algorithms is evaluated based on their Win, Draw, and Loss percentages.

---

## Technologies Used

- Python
- Reinforcement Learning
- Temporal-Difference Learning
- Q-Learning
- SARSA
- Expected SARSA
- n-step TD Learning
- GitHub
- Google Colab

---

## Planning and Learning using Tabular Methods

### Dyna-Q

Dyna-Q combines direct reinforcement learning with model-based planning.

The agent learns from real interactions with the Tic-Tac-Toe environment and stores the observed transitions in a model. It then uses simulated experiences from the model to perform additional planning updates.

The Dyna-Q agent was trained for 50,000 episodes and evaluated using 100 test games.

### Dyna-Q Results

| Algorithm | Win % | Draw % | Loss % |
|---|---:|---:|---:|
| Dyna-Q | 62.00% | 3.00% | 35.00% |

### Dyna-Q Discussion

Dyna-Q achieved a win rate of 62.00% in the conducted experiment. The result demonstrates the application of both learning from real experience and planning from a learned environment model using tabular methods.

## Project Structure

```text
Tic-Tac-Toe-Q-Learning/
│
├── tic_tac_toe.py
│
├── q_learning.py
├── sarsa.py
├── expected_sarsa.py
│
├── n_step_td.py
├── n_step_sarsa.py
├── n_step_off_policy.py
├── n_step_comparison.py
│
├── train.py
├── train_comparison.py
├── evaluate.py
├── evaluate_comparison.py
│
├── README.md
└── requirements.txt
