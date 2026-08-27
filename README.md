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
