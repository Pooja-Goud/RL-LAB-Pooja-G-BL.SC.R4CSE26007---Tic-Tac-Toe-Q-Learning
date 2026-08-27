
import random


class SARSAAgent:

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.01
    ):
        self.q_table = {}

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_q_values(self, state):

        if state not in self.q_table:
            self.q_table[state] = [0.0] * 9

        return self.q_table[state]

    def choose_action(self, state, available_actions):

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(available_actions)

        # Exploitation
        q_values = self.get_q_values(state)

        max_q = max(
            q_values[action]
            for action in available_actions
        )

        best_actions = [
            action
            for action in available_actions
            if q_values[action] == max_q
        ]

        return random.choice(best_actions)

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        next_action,
        done
    ):

        q_values = self.get_q_values(state)

        current_q = q_values[action]

        if done:
            target = reward
        else:
            next_q_values = self.get_q_values(next_state)

            target = (
                reward
                + self.discount_factor
                * next_q_values[next_action]
            )

        q_values[action] = (
            current_q
            + self.learning_rate
            * (target - current_q)
        )

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )
