import random


class ExpectedSARSAAgent:

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

        if random.random() < self.epsilon:
            return random.choice(available_actions)

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

    def expected_value(self, state, available_actions):

        q_values = self.get_q_values(state)

        if not available_actions:
            return 0

        max_q = max(
            q_values[action]
            for action in available_actions
        )

        best_actions = [
            action
            for action in available_actions
            if q_values[action] == max_q
        ]

        probability_random = (
            self.epsilon / len(available_actions)
        )

        probability_best = (
            (1 - self.epsilon) / len(best_actions)
        )

        expected_q = 0

        for action in available_actions:

            probability = probability_random

            if action in best_actions:
                probability += probability_best

            expected_q += (
                probability * q_values[action]
            )

        return expected_q

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        next_available_actions,
        done
    ):

        q_values = self.get_q_values(state)

        current_q = q_values[action]

        if done:
            target = reward

        else:

            expected_next_q = self.expected_value(
                next_state,
                next_available_actions
            )

            target = (
                reward
                + self.discount_factor
                * expected_next_q
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
