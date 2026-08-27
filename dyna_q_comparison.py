from tic_tac_toe import TicTacToe
from dyna_q import DynaQAgent


EPISODES = 50000
TEST_GAMES = 100


def train_dyna_q():

    agent_x = DynaQAgent(
        planning_steps=10
    )

    agent_o = DynaQAgent(
        planning_steps=10
    )

    game = TicTacToe()

    for episode in range(EPISODES):

        game.reset()

        done = False

        while not done:

            current_player = game.current_player

            current_agent = (
                agent_x
                if current_player == "X"
                else agent_o
            )

            state = game.get_state()

            available_actions = (
                game.available_actions()
            )

            action = current_agent.choose_action(
                state,
                available_actions
            )

            game.make_move(action)

            result = game.check_winner()

            if result is not None:

                if result == "Draw":

                    reward = 0

                elif result == current_player:

                    reward = 1

                else:

                    reward = -1

                current_agent.learn(
                    state,
                    action,
                    reward,
                    game.get_state(),
                    []
                )

                done = True

            else:

                game.switch_player()

                next_state = game.get_state()

                next_available_actions = (
                    game.available_actions()
                )

                current_agent.learn(
                    state,
                    action,
                    0,
                    next_state,
                    next_available_actions
                )

        agent_x.decay_epsilon()
        agent_o.decay_epsilon()

        if (episode + 1) % 5000 == 0:

            print(
                f"Training Dyna-Q... "
                f"Episode {episode + 1}/{EPISODES}"
            )

    return agent_x, agent_o


def evaluate(agent_x, agent_o):

    wins = 0
    draws = 0
    losses = 0

    # Disable exploration during testing
    agent_x.epsilon = 0
    agent_o.epsilon = 0

    for _ in range(TEST_GAMES):

        game = TicTacToe()

        done = False

        while not done:

            current_player = (
                game.current_player
            )

            agent = (
                agent_x
                if current_player == "X"
                else agent_o
            )

            state = game.get_state()

            available_actions = (
                game.available_actions()
            )

            action = agent.choose_action(
                state,
                available_actions
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
    print("DYNA-Q TIC-TAC-TOE")
    print("=" * 60)

    print("\nTraining Dyna-Q...")

    agent_x, agent_o = train_dyna_q()

    print("\nTraining completed.")

    print("\nEvaluating Dyna-Q...")

    results = evaluate(
        agent_x,
        agent_o
    )

    print("\nDyna-Q Results")
    print("-" * 40)

    print(
        f"Wins:   {results[0]}"
    )

    print(
        f"Draws:  {results[1]}"
    )

    print(
        f"Losses: {results[2]}"
    )

    print(
        f"Win Percentage:  {results[3]:.2f}%"
    )

    print(
        f"Draw Percentage: {results[4]:.2f}%"
    )

    print(
        f"Loss Percentage: {results[5]:.2f}%"
    )
