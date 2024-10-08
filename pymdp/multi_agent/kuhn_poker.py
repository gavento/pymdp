from pymdp.envs import Env
from typing import Optional
import numpy as np

class KuhnPokerEnv(Env):
    def __init__(self):
        self.deck = ['K', 'Q', 'J']
        self.player_hands = []
        self.pot = 0
        self.action_history = []
        self.turn = 0
        self.reset()
        self._transition_dist = self._construct_transition_dist()
        self._likelihood_dist = self._construct_likelihood_dist()
    
    def reset(self, state=None):
        """
        Resets the game state: shuffles deck, deals cards, resets pot and turn.
        """
        np.random.shuffle(self.deck)  # Use numpy's shuffle method
        self.player_hands = [self.deck[0], self.deck[1]]  # Deal one card to each player
        self.pot = 2  # Both players ante 1
        self.action_history = []
        self.turn = 0  # Player 1's turn
        return self._get_observation()

    def step(self, action):
        """
        Executes the given action and moves the game forward. The action must 
        be valid for the current player.
        
        Returns:
            observation (tuple): The next observation (what the current player sees)
            reward (float): Reward for the current turn (can be 0 until game ends)
            done (bool): Whether the game is over
            info (dict): Any extra information
        """
        # Convert NumPy string to Python string before storing the action
        action = str(action)
        if len(self.action_history) > 0:
            last_action = self.action_history[-1]
        else:
            last_action = None

        # Record the actionl
        self.action_history.append(action)
        done = False
        reward = 0
        
        if self.turn == 0:  # Player 1's turn
            if action == 'bet':
                self.pot += 1  # Player 1 adds to the pot
            elif action == 'check':
                pass
            self.turn = 1  # Switch to Player 2's turn
        elif self.turn == 1:  # Player 2's turn
            if last_action == 'bet':
                if action == 'fold':
                    done = True
                    reward = self._resolve_fold(1)  # Player 1 wins if Player 2 folds
                elif action == 'call':
                    self.pot += 1
                    done = True
                    reward = self._resolve_game()
            elif last_action == 'check':
                if action == 'bet':
                    self.pot += 1
                    self.turn == 2
                elif action == 'check':
                    done = True
                    reward = self._resolve_game()
            # if action == 'bet':
            #     self.pot += 1
            # elif action == 'check':
            #     if self.action_history[-2] == 'check':
            #         done = True
            #         reward = self._resolve_game()
        # check bet
        elif self.turn == 2:
            if action == 'fold':
                done = True
                reward = self._resolve_fold(0)  # Player 2 wins if Player 1 folds
            elif action == 'call':
                self.pot += 1
                done = True
                reward = self._resolve_game()
        # Observation: what the current player sees
        observation = self._get_observation()
        
        return observation, reward, done, {}

    def _get_observation(self):
        """
        Returns the current observation for the active player.
        """
        if self.turn == 0:  # Player 1's observation (only their own hand)
            return {'hand': self.player_hands[0], 'actions': self.action_history}
        elif self.turn == 1:  # Player 2's observation (their hand + Player 1's action)
            return {'hand': self.player_hands[1], 'actions': self.action_history}
        elif self.turn == 2:
            return {'hand': self.player_hands[0], 'actions': self.action_history}
        else:  # After the game ends, both players observe rewards
            return {'hands': self.player_hands, 'actions': self.action_history}
    
    def _resolve_game(self):
        """
        Resolves the game after a showdown and determines the winner.
        """
        # Compare hands: King > Queen > Jack
        rank = {'K': 3, 'Q': 2, 'J': 1}
        if rank[self.player_hands[0]] > rank[self.player_hands[1]]:
            return self.pot  # Player 1 wins
        else:
            return -self.pot  # Player 2 wins
    
    def _resolve_fold(self, folding_player):
        """
        Resolves the game if a player folds.
        """
        if folding_player == 0:
            return -self.pot  # Player 1 folds, Player 2 wins
        elif folding_player == 1:
            return self.pot  # Player 2 folds, Player 1 wins
    
    def render(self):
        """
        Rendering function to display the current state of the game.
        """
        print(f"Player 1's hand: {self.player_hands[0]}")
        print(f"Player 2's hand: {self.player_hands[1]}")
        print(f"Action history: {self.action_history}")
        print(f"Pot: {self.pot}")
    
    def sample_action(self):
        """
        Returns a random valid action for the current player using NumPy.
        """
        if self.turn == 0:  # Player 1's possible actions
            return np.random.choice(['check', 'bet'])
        elif self.turn == 1:  # Player 2's possible actions
            if self.action_history[-1] == 'bet':  # If Player 1 bet, Player 2 can fold or call
                return np.random.choice(['fold', 'call'])
            else:
                return np.random.choice(['check', 'bet'])

    def get_likelihood_dist(self):
        return self._likelihood_dist

    def get_transition_dist(self):
        return self._transition_dist

    def _construct_likelihood_dist(self):
        """ Returns the likelihood distribution A for the Kuhn Poker game. """
        print("test")
        # Define the number of states and observations
        num_hands = 6  # JQ, JK, QJ, QK, KJ, KQ
        num_action_histories = 7  # [], [check], [bet], [check, check], [check, bet], [check, bet, fold], [check, bet, call]
        num_cards = 3  # J, Q, K
        num_prev_actions = 3  # None, Check/Call, Bet/Fold

        # Initialize the likelihood distribution A
        A = np.zeros((num_cards, num_prev_actions, num_hands, num_action_histories))

        # Set the likelihoods for each player's observations
        for hand in range(num_hands):
            for history in range(num_action_histories):
                # Player's card observation
                if hand < 2:  # JQ, JK
                    A[0, :, hand, history] = 1.0  # Player sees J
                elif hand < 4:  # QJ, QK
                    A[1, :, hand, history] = 1.0  # Player sees Q
                else:  # KJ, KQ
                    A[2, :, hand, history] = 1.0  # Player sees K

                # Previous action observation
                if history == 0:  # []
                    A[:, 0, hand, history] = 1.0  # No previous action
                elif history in [1, 3]:  # [check], [check, check]
                    A[:, 1, hand, history] = 1.0  # Previous action was Check
                elif history in [2, 4, 5, 6]:  # [bet], [check, bet], [check, bet, fold], [check, bet, call]
                    A[:, 2, hand, history] = 1.0  # Previous action was Bet

        return A

    def _construct_transition_dist(self):
        print("test")
        """ Returns the transition distribution B for the Kuhn Poker game. """
        # States: (hand, action history)
        # Transition: (hand, action history) -> (hand, (action_history, next action))
        # Define the number of states and actions
        num_hands =  6 # JQ, JK, QJ, QK, KJ, KQ
        num_action_histories = 7  # [], [check], [bet], [check, check], [check, bet], [check, bet, fold], [check, bet, call]
        num_actions = 2  # check/call, bet/fold

        # Initialize the transition distribution B
        B = np.zeros((num_hands, num_action_histories, num_actions, num_hands, num_action_histories))

        # The hands don't change during the game, so we'll set those transitions to 1
        for h in range(num_hands):
            for ah in range(num_action_histories):
                for a in range(num_actions):
                    B[h, ah, a, h, :] = 1.0 / num_action_histories

        # Now we need to set the transitions for the action histories
        # We'll use a dictionary to map current histories to possible next histories
        history_transitions = {
            0: [1, 2],           # [] can transition to [check] or [bet]
            1: [3, 4],           # [check] can transition to [check, check] or [check, bet]
            2: [5, 6],           # [bet] can transition to [bet, fold] or [bet, call]
            3: [],               # [check, check] is terminal
            4: [5, 6],           # [check, bet] can transition to [check, bet, fold] or [check, bet, call]
            5: [],               # [check, bet, fold] is terminal
            6: []                # [check, bet, call] is terminal
        }

        # Set the transitions based on the history_transitions dictionary
        for h in range(num_hands):
            for ah in range(num_action_histories):
                for a in range(num_actions):
                    if history_transitions[ah]:
                        B[h, ah, a, h, history_transitions[ah][a]] = 1.0

        return B

class GeneralKuhnPokerEnv(Env):
    def __init__(self, num_players=3):
        self.num_players = num_players
        self.deck = [f'Card_{i+1}' for i in range(num_players + 1)]  # Generalize to K+1 cards
        self.player_hands = []
        self.pot = 0
        self.action_history = []
        self.turn = 0
        self.active_players = []  # Track active players (those who haven't folded)
        self.current_bet = 0  # Track current bet
        self.reset()

    def reset(self, state=None):
        """
        Resets the game state: shuffles deck, deals cards, resets pot and turn.
        """
        np.random.shuffle(self.deck)  # Shuffle deck
        self.player_hands = self.deck[:self.num_players]  # Deal cards to each player
        self.pot = self.num_players  # Each player antes 1
        self.action_history = []
        self.turn = 0  # Player 1's turn
        self.active_players = list(range(self.num_players))  # All players start active
        self.current_bet = 0  # No outstanding bet at the start
        return self._get_observation()

    def step(self, action):
        """
        Executes the given action and moves the game forward.
        
        Returns:
            observation: The next observation (what the current player sees)
            reward: Reward for the current turn (0 until game ends)
            done: Whether the game is over
            info: Any extra information
        """
        # Convert NumPy string to Python string before storing the action
        action = str(action)
        current_player = self.active_players[self.turn]  # Track current player by their index in active_players
        self.action_history.append((current_player, action))
        done = False
        reward = 0

        # Handle the action logic for bet, check, fold, call
        if action == 'fold':
            self.active_players.remove(current_player)  # Remove player from the active list

        elif action == 'bet':
            self.current_bet += 1
            self.pot += 1  # Increase the pot
        elif action == 'call':
            self.pot += 1  # Player calls and adds to the pot
        # Check doesn't affect the pot or bet amount

        # If only one player remains after any fold, they win the game
        if len(self.active_players) == 1:
            done = True
            reward = self.pot  # Last remaining player wins the pot
        else:
            # Move to the next active player
            self.turn = (self.turn + 1) % len(self.active_players)

            # If all active players have acted (everyone had a turn), check for showdown
            if self.turn == 0 and self._round_is_complete():
                done = True
                reward = self._resolve_game()

        # Observation: what the current player sees
        observation = self._get_observation()
        
        return observation, reward, done, {}

    def _round_is_complete(self):
        """
        Check if all players have acted in this round. This function ensures that after every player has had their turn,
        we either proceed to showdown or end the game.
        """
        # In each round, all active players should have acted once
        last_round_actions = self.action_history[-len(self.active_players):]
        last_round_players = [action[0] for action in last_round_actions]
        return set(last_round_players) == set(self.active_players)

    def _get_observation(self):
        """
        Returns the current observation for the active player.
        """
        current_player = self.turn
        actions_so_far = self.action_history
        return {
            'hand': self.player_hands[current_player],  # Current player's hand
            'actions': actions_so_far  # Actions of previous players
        }

    def _check_for_showdown(self):
        """
        If all players have acted, check if it's time for a showdown or if the game continues.
        """
        if all(action == 'check' for _, action in self.action_history[-self.num_players:]):
            # All players checked, time for a showdown
            return True
        return False
    
    def _resolve_game(self):
        """
        Resolves the game after all players have taken their turn.
        """
        # If there's only one active player left (due to folds), they win the pot
        if len(self.active_players) == 1:
            winner = self.active_players[0]
            print(f"Player {winner + 1} wins by default (all others folded)")
            return self.pot  # Winner takes the entire pot
        
        # Otherwise, resolve based on hands (showdown)
        rank = {'Card_1': 1, 'Card_2': 2, 'Card_3': 3, 'Card_4': 4}  # Simplified ranking logic for 3-player
        best_player = max(self.active_players, key=lambda p: rank[self.player_hands[p]])
        print(f"Player {best_player + 1} wins with hand {self.player_hands[best_player]}")
        return self.pot  # Winner takes the entire pot

    def render(self):
        """
        Rendering function to display the current state of the game.
        """
        print(f"Player hands: {self.player_hands}")
        print(f"Action history: {self.action_history}")
        print(f"Pot: {self.pot}")
        print(f"Active players: {self.active_players}")
    
    def sample_action(self):
        """
        Returns a random valid action for the current player using NumPy.
        """
        if len(self.action_history) == 0 or self.action_history[-1][1] != 'bet':
            # If no bet has been placed or the last action was not 'bet', the player can check or bet
            return np.random.choice(['check', 'bet'])
        else:
            # If the last action was 'bet', the player can fold or call
            return np.random.choice(['fold', 'call'])


