from pymdp.multi_agent.kuhn_poker import KuhnPokerEnv

def play_kuhn_poker():
    # Initialize the Kuhn Poker environment
    env = KuhnPokerEnv()
    
    # Reset the environment and get the initial observation
    observation = env.reset()
    print(f"Initial observation: {observation}")
    
    done = False
    round_number = 1
    
    # Play through the game until it is done
    while not done:
        print(f"\n--- Round {round_number} ---")
        
        # Player 1 takes an action
        if env.turn == 0:
            action = env.sample_action()  # Random action for Player 1
            print(f"Player 1 chooses to {action}")
        
        # Player 2 takes an action after Player 1
        elif env.turn == 1:
            action = env.sample_action()  # Random action for Player 2
            print(f"Player 2 chooses to {action}")
        
        # Perform the action and update the game state
        observation, reward, done, info = env.step(action)
        print(f"Observation after action: {observation}")
        
        if done:
            print(f"Game over! Final reward: {reward}")
        round_number += 1

    # Render the final state of the game
    env.render()

# Run the game
if __name__ == "__main__":
    play_kuhn_poker()
