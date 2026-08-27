from tic_tac_toe import TicTacToe
from q_learning import QLearningAgent
from train import train


def human_move(game):

    while True:

        try:
            position = int(
                input("Enter your move (1-9): ")
            ) - 1

            if position in game.available_actions():
                return position

            print("Invalid move. Try again.")

        except ValueError:
            print("Please enter a number from 1 to 9.")


def play_against_agent(agent):

    game = TicTacToe()

    print("\nYou are X.")
    print("The AI is O.")

    while True:

        game.display()

        if game.current_player == "X":

            action = human_move(game)

        else:

            state = game.get_state()

            action = agent.choose_action(
                state,
                game.available_actions()
            )

            print(f"AI chooses position {action + 1}")

        game.make_move(action)

        result = game.check_winner()

        if result is not None:

            game.display()

            if result == "Draw":
                print("Game Draw!")

            elif result == "X":
                print("You Win!")

            else:
                print("AI Wins!")

            break

        game.switch_player()


if __name__ == "__main__":

    print("Training AI...")

    agent_x, agent_o = train()

    print("\nTraining completed.")

    # AI plays as O
    play_against_agent(agent_o)
