import torch
import random
import numpy as np
from snake import SnakeGameAI, Direction, Point  # (we'll tweak snake.py slightly for AI control)
from agent import Agent
from plot import plot  # Optional: for live graph plotting

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # 1. Get the old state
        state_old = agent.get_state(game)

        # 2. Get move based on the old state
        final_move = agent.get_action(state_old)

        # 3. Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # 4. Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 5. Remember (store in memory)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # Save model (optional)
                torch.save(agent.model.state_dict(), 'model.pth')

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
