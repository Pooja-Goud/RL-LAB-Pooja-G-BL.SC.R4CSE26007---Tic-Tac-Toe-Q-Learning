## Performance Comparison

Each algorithm was trained for 50,000 episodes and evaluated using 100 test games.

### Q-Learning, SARSA and Expected SARSA

| Algorithm | Win % | Draw % | Loss % |
|---|---:|---:|---:|
| Q-Learning | 66.00% | 1.00% | 33.00% |
| SARSA | 71.00% | 5.00% | 24.00% |
| Expected SARSA | 69.00% | 3.00% | 28.00% |

### n-step Methods

The n-step experiments used a 3-step return.

| Algorithm | Win % | Draw % | Loss % |
|---|---:|---:|---:|
| 3-step SARSA | 55.00% | 16.00% | 29.00% |
| 3-step Off-Policy | 56.00% | 16.00% | 28.00% |

## Results and Discussion

The experimental results show that SARSA achieved the highest win percentage among the implemented algorithms, with a win rate of 71.00% and a loss rate of 24.00%.

Expected SARSA achieved a win rate of 69.00%, while Q-Learning achieved 66.00%.

Among the n-step methods, 3-step Off-Policy achieved a slightly higher win rate of 56.00% compared with 55.00% for 3-step SARSA.

These results are specific to the experimental setup and evaluation runs used in this project. The performance of reinforcement learning algorithms may vary depending on training parameters, random initialization, exploration strategy, and the number of evaluation games.

## Conclusion

This project demonstrates the implementation and comparison of several reinforcement learning algorithms for Tic-Tac-Toe.

The implemented methods include:

- Q-Learning
- SARSA
- Expected SARSA
- n-step SARSA
- n-step Off-Policy Learning

The experiments demonstrate how different Temporal-Difference learning approaches can produce different performance when applied to the same game environment.
