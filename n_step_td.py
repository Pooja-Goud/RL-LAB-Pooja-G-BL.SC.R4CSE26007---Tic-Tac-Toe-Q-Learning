class NStepTDPrediction:

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.9,
        n=3
    ):
        self.values = {}

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.n = n

    def get_value(self, state):

        if state not in self.values:
            self.values[state] = 0.0

        return self.values[state]

    def update(
        self,
        states,
        rewards
    ):

        # states:
        # S0, S1, S2, ...

        # rewards:
        # R1, R2, R3, ...

        state = states[0]

        G = 0.0

        steps = min(
            self.n,
            len(rewards)
        )

        for i in range(steps):

            G += (
                self.discount_factor ** i
            ) * rewards[i]

        # If the episode has not ended,
        # bootstrap from the n-step state.

        if len(states) > self.n:

            next_state = states[self.n]

            G += (
                self.discount_factor ** self.n
            ) * self.get_value(next_state)

        current_value = self.get_value(state)

        self.values[state] = (
            current_value
            + self.learning_rate
            * (G - current_value)
        )

    def get_state_value(self, state):

        return self.get_value(state)
