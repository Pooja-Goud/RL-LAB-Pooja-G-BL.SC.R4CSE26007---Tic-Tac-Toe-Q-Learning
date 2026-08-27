from tic_tac_toe import TicTacToe
from q_learning import QLearningAgent
from sarsa import SARSAAgent
from expected_sarsa import ExpectedSARSAAgent


EPISODES = 50000


def train_q_learning():
    agent_x = QLearningAgent()
    agent_o = QLearningAgent()

    game = TicTacToe()

    for episode in range(EPISODES):
        game.reset()
        done = False

        while not done:
            current_player = game.current_player
            agent = agent_x if current_player == "X" else agent_o

            state = game.get_state()
            actions = game.available_actions()

            action = agent.choose_action(state, actions)

            game.make_move(action)
            result = game.check_winner()

            if result is not None:
                if result == "Draw":
                    reward = 0
                elif result == current_player:
                    reward = 1
                else:
                    reward = -1

                agent.update(
                    state,
                    action,
                    reward,
                    game.get_state(),
                    [],
                    True
                )

                done = True

            else:
                game.switch_player()

                agent.update(
                    state,
                    action,
                    0,
                    game.get_state(),
                    game.available_actions(),
                    False
                )

        agent_x.decay_epsilon()
        agent_o.decay_epsilon()

    return agent_x, agent_o


def train_sarsa():
    agent_x = SARSAAgent()
    agent_o = SARSAAgent()

    game = TicTacToe()

    for episode in range(EPISODES):
        game.reset()
        done = False

        state = game.get_state()
        actions = game.available_actions()

        current_agent = agent_x

        action = current_agent.choose_action(state, actions)

        while not done:
            current_player = game.current_player

            current_agent = (
                agent_x if current_player == "X" else agent_o
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

                current_agent.update(
                    state,
                    action,
                    reward,
                    game.get_state(),
                    None,
                    True
                )

                done = True

            else:
                game.switch_player()

                next_state = game.get_state()
                next_actions = game.available_actions()

                next_agent = (
                    agent_x
                    if game.current_player == "X"
                    else agent_o
                )

                next_action = next_agent.choose_action(
                    next_state,
                    next_actions
                )

                current_agent.update(
                    state,
                    action,
                    0,
                    next_state,
                    next_action,
                    False
                )

                state = next_state
                action = next_action

        agent_x.decay_epsilon()
        agent_o.decay_epsilon()

    return agent_x, agent_o


def train_expected_sarsa():
    agent_x = ExpectedSARSAAgent()
    agent_o = ExpectedSARSAAgent()

    game = TicTacToe()

    for episode in range(EPISODES):
        game.reset()
        done = False

        while not done:
            current_player = game.current_player

            current_agent = (
                agent_x if current_player == "X" else agent_o
            )

            state = game.get_state()
            actions = game.available_actions()

            action = current_agent.choose_action(
                state,
                actions
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

                current_agent.update(
                    state,
                    action,
                    reward,
                    game.get_state(),
                    [],
                    True
                )

                done = True

            else:
                game.switch_player()

                next_state = game.get_state()
                next_actions = game.available_actions()

                current_agent.update(
                    state,
                    action,
                    0,
                    next_state,
                    next_actions,
                    False
                )

        agent_x.decay_epsilon()
        agent_o.decay_epsilon()

    return agent_x, agent_o


if __name__ == "__main__":

    print("Training Q-Learning...")
    train_q_learning()
    print("Q-Learning training completed.")

    print("\nTraining SARSA...")
    train_sarsa()
    print("SARSA training completed.")

    print("\nTraining Expected SARSA...")
    train_expected_sarsa()
    print("Expected SARSA training completed.")

    print("\nAll training completed successfully!")
