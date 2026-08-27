import random


class NStepSARSAAgent:

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.01,
        n=3
    ):

        self.q_table = {}

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.n = n

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
        states,
        actions,
        rewards
    ):

        """
        n-step SARSA update.

        states:
            S0, S1, S2, ...

        actions:
            A0, A1, A2, ...

        rewards:
            R1, R2, R3, ...
        """

        if not states or not actions:
            return

        state = states[0]
        action = actions[0]

        G = 0.0

        steps = min(
            self.n,
            len(rewards)
        )

        # Calculate n-step return
        for i in range(steps):

            G += (
                self.discount_factor ** i
            ) * rewards[i]

        # Bootstrap using Q(Sn, An)
        if (
            len(states) > self.n
            and len(actions) > self.n
        ):

            next_state = states[self.n]
            next_action = actions[self.n]

            next_q = self.get_q_values(
                next_state
            )[next_action]

            G += (
                self.discount_factor ** self.n
            ) * next_q

        current_q = self.get_q_values(
            state
        )[action]

        # Update Q-value
        self.get_q_values(state)[action] = (
            current_q
            + self.learning_rate
            * (G - current_q)
        )

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )
