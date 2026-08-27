from tic_tac_toe import TicTacToe
from q_learning import QLearningAgent


EPISODES = 50000


def train():

    agent_x = QLearningAgent()
    agent_o = QLearningAgent()

    game = TicTacToe()

    for episode in range(EPISODES):

        game.reset()

        done = False

        while not done:

            current_player = game.current_player

            if current_player == "X":
                agent = agent_x
            else:
                agent = agent_o

            state = game.get_state()

            available_actions = game.available_actions()

            action = agent.choose_action(
                state,
                available_actions
            )

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

                agent.update(
                    state,
                    action,
                    reward,
                    game.get_state(),
                    [],
                    True
                )

            else:

                game.switch_player()

                next_state = game.get_state()

                next_actions = game.available_actions()

                agent.update(
                    state,
                    action,
                    0,
                    next_state,
                    next_actions,
                    False
                )

        agent_x.decay_epsilon()
        agent_o.decay_epsilon()

        if (episode + 1) % 5000 == 0:
            print(
                f"Episode {episode + 1}/{EPISODES} "
                f"| Epsilon X: {agent_x.epsilon:.4f} "
                f"| Epsilon O: {agent_o.epsilon:.4f}"
            )

    return agent_x, agent_o


if __name__ == "__main__":

    agent_x, agent_o = train()

    print("\nTraining completed!")
    print(f"Q-table size for X: {len(agent_x.q_table)}")
    print(f"Q-table size for O: {len(agent_o.q_table)}")
