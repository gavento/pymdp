from pymdp.multi_agent.kuhn_poker import GeneralKuhnPokerEnv

def play_general_kuhn_poker(num_players):
    # Initialize the Kuhn Poker environment with K players
    env = GeneralKuhnPokerEnv(num_players=num_players)
    
    # Reset the environment and get the initial observation
    observation = env.reset()
    print(f"Initial observation: {observation}")
    
    done = False
    round_number = 1
    
    # Play through the game until it is done
    while not done:
        print(f"\n--- Round {round_number} ---")
        
        # Current player takes an action
        action = env.sample_action()  # Random action
        print(f"Player {env.turn + 1} chooses to {action}")
        
        # Perform the action and update the game state
        observation, reward, done, info = env.step(action)
        print(f"Observation after action: {observation}")
        
        if done:
            print(f"Game over! Final reward: {reward}")
        round_number += 1

    # Render the final state of the game
    env.render()

# Run the game with K players
if __name__ == "__main__":
    K = 3  # Example with 3 players
    play_general_kuhn_poker(K)
