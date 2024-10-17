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
        self.num_hands = 6
        self.num_cards = 3
        self.num_action_histories = 9
        self.num_prev_actions = 5
        self.num_rewards = 4
        self.reset()
        self._transition_dist = self._construct_transition_dist()
        self._likelihood_dist = self._construct_likelihood_dist()
        # self._hist_idx_dict = self._construct_hist_idx_dict()
        # self._prev_action_idx_dict = self._construct_prev_action_idx_dict()
    
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
            self.pot = 2 # both players ante 1
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
    
    def _construct_hist_idx_dict(self):
        """
        Constructs a dictionary mapping action histories to their corresponding indices.
        """
        hist_idx_dict = {}
        i = 0
        for hist in enumerate(self._idx_to_hist(range(9))):
            hist_idx_dict[i] = hist
            i += 1
        print(hist_idx_dict)
        return hist_idx_dict

    def _construct_prev_action_idx_dict(self):
        """
        Constructs a dictionary mapping previous actions to their corresponding indices.
        """
        prev_action_idx_dict = {}
        i = 0
        for prev_action in enumerate(self._idx_to_prev_action(range(5))):
            prev_action_idx_dict[i] = prev_action
            i += 1
        print(prev_action_idx_dict)
        return prev_action_idx_dict
    
    def _hist_to_idx(self, history):
        """
        Converts the action history to its corresponding index.
        """
        if history == []:
            return 0
        elif history == ['check']:
            return 1
        elif history == ['bet']:
            return 2
        elif history == ['check', 'check']:
            return 3        
        elif history == ['check', 'bet']:
            return 4
        elif history == ['bet', 'fold']:
            return 5
        elif history == ['bet', 'call']:
            return 6
        elif history == ['check', 'bet', 'fold']:
            return 7
        elif history == ['check', 'bet', 'call']:
            return 8
    
    def _idx_to_hist(self, idx):
        """
        Converts the index of an action history to its corresponding array representation.
        """
        if idx == 0:
            return []
        elif idx == 1:
            return ['check']
        elif idx == 2:
            return ['bet']
        elif idx == 3:
            return ['check', 'check']
        elif idx == 4:
            return ['check', 'bet']
        elif idx == 5:
            return ['bet', 'fold']
        elif idx == 6:
            return ['bet', 'call']
        elif idx == 7:
            return ['check', 'bet', 'fold']
        elif idx == 8:
            return ['check', 'bet', 'call']

    def _idx_to_prev_action(self, idx):
        """
        Converts the index of an action to its corresponding string representation.
        """
        if idx == 0:
            return None
        elif idx == 1:
            return 'check'
        elif idx == 2:
            return 'bet'
        elif idx == 3:
            return 'call'
        elif idx == 4:
            return 'fold'
    
    def _prev_action_to_idx(self, prev_action):
        """
        Converts the string representation of a previous action to its corresponding index.
        """
        if prev_action is None:
            return 0
        elif prev_action == 'check':
            return 1
        elif prev_action == 'bet':
            return 2
        elif prev_action == 'call':
            return 3
        elif prev_action == 'fold':
            return 4    

    def _hand_to_idx(self, hand):
        """
        Converts the hand to its corresponding index.
        """
        if hand == ['J', 'Q']:
            return 0
        elif hand == ['J', 'K']:
            return 1
        elif hand == ['Q', 'J']:
            return 2
        elif hand == ['Q', 'K']:
            return 3
        elif hand == ['K', 'J']:
            return 4
        elif hand == ['K', 'Q']:
            return 5
    
    def _idx_to_hand(self, idx):
        """
        Converts the index of a hand to its corresponding array representation.
        """
        if idx == 0:
            return ['J', 'Q']
        elif idx == 1:
            return ['J', 'K']
        elif idx == 2:
            return ['Q', 'J']
        elif idx == 3:
            return ['Q', 'K']
        elif idx == 4:
            return ['K', 'J']
        elif idx == 5:
            return ['K', 'Q']

    def _card_to_idx(self, card):
        """
        Converts the card to its corresponding index.
        """
        if card == 'J':
            return 0
        elif card == 'Q':
            return 1
        elif card == 'K':
            return 2

    def _idx_to_card(self, idx):
        """
        Converts the index of a card to its corresponding array representation.
        """
        if idx == 0:
            return 'J'
        elif idx == 1:
            return 'Q'
        elif idx == 2:
            return 'K'
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

    def _construct_likelihood_dist(self, player_idx):
        """ Returns the likelihood distribution A for the Kuhn Poker game. """
        # Define the number of states and observations
        # num_hands = 6  # JQ, JK, QJ, QK, KJ, KQ
        # num_action_histories = 9  # [], [check], [bet], [check, check], [check, bet], [bet, fold], [bet, call], [check, bet, fold], [check, bet, call]
        # num_cards = 3  # J, Q, K
        # num_prev_actions = 5  # None, Check, Bet, Call, Fold
        # num_rewards = 4 # -2, -1, 1, 2 - how much each player wins or loses on net

        # state: (hands, action history). observation: (card, previous action)
        # p(o_cards | hidden states) and p(o_actions | hidden states) and p(o_rewards | hidden states)
        # 3 outcome modalities - diff elements of a list
        # create collection of 2 A matrices - one for cards and one for actions - each A matrix is representing the likelihood over that modality
        # leading dimension of matrices stores the support of the distribution - A_0: 3 rows, A_1: 3 rows
        # currently assuming posterior over hands and action histories are independent - we don't want this
        # use a single hidden state factor to represent both hands and action histories
        # could fill out like below and then reshape at the end
        # A lagging dim to be 54 for both cards and actions
        # for each column in one of the A matrices, fill out the probabilities
        
        # Initialize the likelihood distribution A
        A_cards = np.zeros((self.num_cards, self.num_hands, self.num_action_histories))
        A_actions = np.zeros((self.num_prev_actions, self.num_hands, self.num_action_histories))
        A_rewards = np.zeros((self.num_rewards, self.num_hands, self.num_action_histories))
        # Fill in A_cards
        for hand_idx in range(self.num_hands):
            for history_idx in range(self.num_action_histories):
                    card_idx = self._card_to_idx(self._idx_to_hand(hand_idx)[0])
                    A_cards[card_idx, hand_idx, history_idx] = 1.0
                    # if hand < 2:  # JQ, JK
                    #     A_cards[0, hand, history] = 1.0  # J
                    # elif hand < 4:  # QJ, QK
                    #     A_cards[1, hand, history] = 1.0  # Q
                    # else:  # KJ, KQ
                    #     A_cards[2, hand, history] = 1.0  # K
                # figure out what the last action was for this history
                last_action = self._prev_action_to_idx(self._idx_to_hist(history_idx)[-1])
                A_actions[last_action, hand_idx, history_idx] = 1.0

                # if history == 0: # None
                #     last_action = 0
                # elif history == 1 or history == 3: # Check
                #     last_action = 1
                # elif history == 2 or history == 4: # Bet
                #     last_action = 2
                # elif history == 5 or history == 7: # Fold
                #     last_action = 3
                # elif history == 6 or history == 8: # Call
                #     last_action = 4


                # figure out the reward for this history
                terminal_histories = [3, 5, 6, 7, 8]
                if history in terminal_histories:
                    

    def _construct_transition_dist(self):
        """ Returns the transition distribution B for the Kuhn Poker game. """
        # States: (hand, action history)
        # Transition: (hand, action history) -> (hand, (action_history, next action))
        # Define the number of states and actions
        # num_hands =  6 # JQ, JK, QJ, QK, KJ, KQ
        # num_action_histories = 7  # [], [check], [bet], [check, check], [check, bet], [check, bet, fold], [check, bet, call]
        # num_actions = 2  # check/call, bet/fold

        # single tensor whose rows and columns are 42, 3rd dim is number of actions - 42 x 42 x 2
        # create dict for hand,action history -> index

        # Initialize the transition distribution B
        B = np.zeros((self.num_hands, self.num_action_histories, self.num_actions, self.num_hands, self.num_action_histories))

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


