from tic_tac_toe import TicTacToe
from q_learning import QLearningAgent
from sarsa import SARSAAgent
from expected_sarsa import ExpectedSARSAAgent

from train_comparison import (
    train_q_learning,
    train_sarsa,
    train_expected_sarsa
)


TEST_GAMES = 100


def evaluate_agent(agent_x, agent_o, games=TEST_GAMES):

    wins = 0
    draws = 0
    losses = 0

    for _ in range(games):

        game = TicTacToe()
        done = False

        # Evaluation should use learned knowledge,
        # not random exploration.
        agent_x.epsilon = 0
        agent_o.epsilon = 0

        while not done:

            current_player = game.current_player

            agent = (
                agent_x
                if current_player == "X"
                else agent_o
            )

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
                    draws += 1

                elif result == "X":
                    wins += 1

                else:
                    losses += 1

            else:
                game.switch_player()

    win_percentage = (wins / games) * 100
    draw_percentage = (draws / games) * 100
    loss_percentage = (losses / games) * 100

    return (
        wins,
        draws,
        losses,
        win_percentage,
        draw_percentage,
        loss_percentage
    )


def print_results(name, results):

    wins, draws, losses, win_pct, draw_pct, loss_pct = results

    print(f"\n{name}")
    print("-" * 40)

    print(f"Wins:   {wins}")
    print(f"Draws:  {draws}")
    print(f"Losses: {losses}")

    print(f"Win Percentage:   {win_pct:.2f}%")
    print(f"Draw Percentage:  {draw_pct:.2f}%")
    print(f"Loss Percentage:  {loss_pct:.2f}%")


if __name__ == "__main__":

    print("=" * 50)
    print("TIC-TAC-TOE REINFORCEMENT LEARNING COMPARISON")
    print("=" * 50)

    # ------------------------------------------------
    # Q-Learning
    # ------------------------------------------------

    print("\nTraining Q-Learning...")
    q_x, q_o = train_q_learning()

    print("Q-Learning training completed.")

    q_results = evaluate_agent(q_x, q_o)

    print_results(
        "Q-Learning Results",
        q_results
    )

    # ------------------------------------------------
    # SARSA
    # ------------------------------------------------

    print("\nTraining SARSA...")
    sarsa_x, sarsa_o = train_sarsa()

    print("SARSA training completed.")

    sarsa_results = evaluate_agent(
        sarsa_x,
        sarsa_o
    )

    print_results(
        "SARSA Results",
        sarsa_results
    )

    # ------------------------------------------------
    # Expected SARSA
    # ------------------------------------------------

    print("\nTraining Expected SARSA...")
    expected_x, expected_o = train_expected_sarsa()

    print("Expected SARSA training completed.")

    expected_results = evaluate_agent(
        expected_x,
        expected_o
    )

    print_results(
        "Expected SARSA Results",
        expected_results
    )

    # ------------------------------------------------
    # Final comparison
    # ------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print(
        f"{'Algorithm':<20}"
        f"{'Win %':<12}"
        f"{'Draw %':<12}"
        f"{'Loss %':<12}"
    )

    print("-" * 60)

    print(
        f"{'Q-Learning':<20}"
        f"{q_results[3]:<12.2f}"
        f"{q_results[4]:<12.2f}"
        f"{q_results[5]:<12.2f}"
    )

    print(
        f"{'SARSA':<20}"
        f"{sarsa_results[3]:<12.2f}"
        f"{sarsa_results[4]:<12.2f}"
        f"{sarsa_results[5]:<12.2f}"
    )

    print(
        f"{'Expected SARSA':<20}"
        f"{expected_results[3]:<12.2f}"
        f"{expected_results[4]:<12.2f}"
        f"{expected_results[5]:<12.2f}"
    )
