from tic_tac_toe import TicTacToe
from n_step_sarsa import NStepSARSAAgent
from n_step_off_policy import NStepOffPolicyAgent


EPISODES = 50000
TEST_GAMES = 100


def train_n_step_sarsa():

    agent_x = NStepSARSAAgent(n=3)
    agent_o = NStepSARSAAgent(n=3)

    game = TicTacToe()

    for episode in range(EPISODES):

        game.reset()

        states = [game.get_state()]
        actions = [
            agent_x.choose_action(
                game.get_state(),
                game.available_actions()
            )
        ]

        rewards = []

        done = False

        while not done:

            current_player = game.current_player

            current_agent = (
                agent_x
                if current_player == "X"
                else agent_o
            )

            state = game.get_state()

            action = actions[-1]

            game.make_move(action)

            result = game.check_winner()

            if result is not None:

                done = True

                if result == "Draw":
                    reward = 0

                elif result == current_player:
                    reward = 1

                else:
                    reward = -1

                rewards.append(reward)

                current_agent.update(
                    states,
                    actions,
                    rewards
                )

            else:

                rewards.append(0)

                game.switch_player()

                next_state = game.get_state()

                next_agent = (
                    agent_x
                    if game.current_player == "X"
                    else agent_o
                )

                next_action = next_agent.choose_action(
                    next_state,
                    game.available_actions()
                )

                states.append(next_state)
                actions.append(next_action)

                current_agent.update(
                    states,
                    actions,
                    rewards
                )

        agent_x.decay_epsilon()
        agent_o.decay_epsilon()

    return agent_x, agent_o


def train_n_step_off_policy():

    agent_x = NStepOffPolicyAgent(n=3)
    agent_o = NStepOffPolicyAgent(n=3)

    game = TicTacToe()

    for episode in range(EPISODES):

        game.reset()

        states = [game.get_state()]
        actions = [
            agent_x.choose_action(
                game.get_state(),
                game.available_actions()
            )
        ]

        rewards = []

        done = False

        while not done:

            current_player = game.current_player

            current_agent = (
                agent_x
                if current_player == "X"
                else agent_o
            )

            state = game.get_state()

            action = actions[-1]

            game.make_move(action)

            result = game.check_winner()

            if result is not None:

                done = True

                if result == "Draw":
                    reward = 0

                elif result == current_player:
                    reward = 1

                else:
                    reward = -1

                rewards.append(reward)

                current_agent.update(
                    states,
                    actions,
                    rewards,
                    game.available_actions()
                )

            else:

                rewards.append(0)

                game.switch_player()

                next_state = game.get_state()

                next_agent = (
                    agent_x
                    if game.current_player == "X"
                    else agent_o
                )

                next_action = next_agent.choose_action(
                    next_state,
                    game.available_actions()
                )

                states.append(next_state)
                actions.append(next_action)

                current_agent.update(
                    states,
                    actions,
                    rewards,
                    game.available_actions()
                )

        agent_x.decay_epsilon()
        agent_o.decay_epsilon()

    return agent_x, agent_o


def evaluate_agent(agent_x, agent_o):

    wins = 0
    draws = 0
    losses = 0

    agent_x.epsilon = 0
    agent_o.epsilon = 0

    for _ in range(TEST_GAMES):

        game = TicTacToe()
        done = False

        while not done:

            current_player = game.current_player

            agent = (
                agent_x
                if current_player == "X"
                else agent_o
            )

            state = game.get_state()

            action = agent.choose_action(
                state,
                game.available_actions()
            )

            game.make_move(action)

            result = game.check_winner()

            if result is not None:

                done = True

                if result == "Draw":
                    draws += 1

                elif result == "X":
                    wins += 1

                else:
                    losses += 1

            else:

                game.switch_player()

    return (
        wins,
        draws,
        losses,
        wins / TEST_GAMES * 100,
        draws / TEST_GAMES * 100,
        losses / TEST_GAMES * 100
    )


if __name__ == "__main__":

    print("=" * 60)
    print("N-STEP REINFORCEMENT LEARNING COMPARISON")
    print("=" * 60)

    print("\nTraining 3-step SARSA...")

    sarsa_x, sarsa_o = train_n_step_sarsa()

    sarsa_results = evaluate_agent(
        sarsa_x,
        sarsa_o
    )

    print("\n3-step SARSA Results")
    print("-" * 40)

    print(
        f"Wins:   {sarsa_results[0]}"
    )

    print(
        f"Draws:  {sarsa_results[1]}"
    )

    print(
        f"Losses: {sarsa_results[2]}"
    )

    print(
        f"Win Percentage:  {sarsa_results[3]:.2f}%"
    )

    print(
        f"Draw Percentage: {sarsa_results[4]:.2f}%"
    )

    print(
        f"Loss Percentage: {sarsa_results[5]:.2f}%"
    )

    print("\nTraining 3-step Off-Policy...")

    off_x, off_o = train_n_step_off_policy()

    off_results = evaluate_agent(
        off_x,
        off_o
    )

    print("\n3-step Off-Policy Results")
    print("-" * 40)

    print(
        f"Wins:   {off_results[0]}"
    )

    print(
        f"Draws:  {off_results[1]}"
    )

    print(
        f"Losses: {off_results[2]}"
    )

    print(
        f"Win Percentage:  {off_results[3]:.2f}%"
    )

    print(
        f"Draw Percentage: {off_results[4]:.2f}%"
    )

    print(
        f"Loss Percentage: {off_results[5]:.2f}%"
    )

    print("\n")
    print("=" * 60)
    print("FINAL N-STEP COMPARISON")
    print("=" * 60)

    print(
        f"{'Algorithm':<25}"
        f"{'Win %':<10}"
        f"{'Draw %':<10}"
        f"{'Loss %':<10}"
    )

    print("-" * 60)

    print(
        f"{'3-step SARSA':<25}"
        f"{sarsa_results[3]:<10.2f}"
        f"{sarsa_results[4]:<10.2f}"
        f"{sarsa_results[5]:<10.2f}"
    )

    print(
        f"{'3-step Off-Policy':<25}"
        f"{off_results[3]:<10.2f}"
        f"{off_results[4]:<10.2f}"
        f"{off_results[5]:<10.2f}"
    )
