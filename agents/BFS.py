from .agent import Agent
import torch
import numpy as np
from cogrid.envs.overcooked.overcooked import Overcooked
from collections import deque
import time

"""
GLOBAL OBSERVATION FEATURE INDEX MAP

Total size: 101 + 2 * (num_agents - 1)

0-3     AgentDir (4)
        one-hot facing direction [right, down, left, up]
4-6     OvercookedInventory (3)
        held object one-hot [onion, soup, plate]
7-10    NextToCounter (4)
        adjacency flags for counters in 4 directions
11-26   NextToPot (16)
        OHE of [empty, <3 onions, cooking, cooked] x 4 for each direction
27-34   Closest Onion (n=4) -> (8)
        (dy, dx) * 4
35-42   Closest Plate (n=4) -> (8)
        (dy, dx) * 4
43-46   Closest PlateStack (n=2) -> (4)
        (dy, dx) * 2
47-50   Closest OnionStack (n=2) -> (4)
        (dy, dx) * 2
51-58   Closest OnionSoup (n=4) -> (8)
        (dy, dx) * 4
59-62   Closest DeliveryZone (n=2) -> (4)
        (dy, dx) * 2
63-70   Closest Counter (n=4) -> (8)
        (dy, dx) * 4
71-92   NClosestPotFeatures (2 pots x 12 features each)
        For each pot k:
        pot_k_reachable
        pot_k_status_ready
        pot_k_status_empty
        pot_k_status_cooking
        pot_k_status_full
        pot_k_contents (# onions in pot)
        pot_k_cooking_timer (0 = finished, -1=not cooking)
        pot_k_dy
        pot_k_dx
        pot_k_row
        pot_k_col
93-94   NClosestPotFeatures (spillover x 2)
        extra slot bc weird cogrid update/bug
95 - 94 + (num_agents - 1) * 2 Distance To Other Players (2 x num_other_players-1)
        (dy, dx) x (num_other_players-1)
(95 + (num_agents - 1) * 2) to (96 + (num_agents - 1) * 2)   AgentPosition (2)
        (row, col)
(97 + (num_agents - 1) * 2) to (100 + (num_agents - 1) * 2)   CanMoveDirection (4)
        movement availability directions
"""

NUM_ONIONS_FOR_FULL = 3 # TODO: dont hardcode

class BFS(Agent):
    def __init__(
        self, 
        env, 
        num_agents,
        save_path=None, 
        log_dir=None, 
        log=False, 
        args=None
    ):
        super().__init__(env, num_agents, save_path, log_dir, log, args)
        self.goal = None
        self.pos = (0, 0)
        self.map = self.build_map()
        self.goal_action = 'wait'
        self.action_map = {
            'up': 0,
            'down': 1,
            'left': 2,
            'right': 3,
            'wait': 6,
            'toggle': 5,
            'pickup': 4,
            'drop': 4, # same as pickup
            
        }
        
        
    def build_map(self):
        if type(self.env) == Overcooked:
            grid = self.env.grid
        else:
            grid = self.env.vec_envs[0].par_env.grid
        rows, cols = grid.height, grid.width
        map = np.ones((rows, cols), dtype=bool)

        for obj in grid.grid:
            if obj is not None:
                r, c = obj.pos
                map[r, c] = False

        return map
        
    def update_position(self, obs):
        self.pos = (int(obs[95 + (self.num_agents - 1) * 2]), int(obs[96 + (self.num_agents - 1) * 2]))
        
    def held_object(self, obs):
        if obs[4]:
            return 'onion'
        elif obs[5]:
            return 'soup'
        elif obs[6]:
            return 'plate'
        else:
            return None
        
    def nearest_cooked_soup(self, obs):
        # cooked and reachable
        if obs[71] and obs[72]:
            return (int(obs[80]), int(obs[81]))
        elif obs[82] and int(obs[83]):
            return (int(obs[91]), int(obs[92]))
        return None
    
    def nearest_cooking_soup(self, obs):
        # cooking and reachable
        if obs[71] and obs[74]:
            return (int(obs[80]), int(obs[81]))
        elif obs[82] and obs[85]:
            return (int(obs[91]), int(obs[92]))
        return None
    
    def nearest_full_soup(self, obs):
        # reachable, full, and not cooking
        if obs[71] and obs[76] >= NUM_ONIONS_FOR_FULL and not obs[74]:
            return (int(obs[80]), int(obs[81]))
        elif obs[82] and obs[87] >= NUM_ONIONS_FOR_FULL and not obs[85]:
            return (int(obs[91]), int(obs[92]))
        return None
    
    def nearest_not_full_soup(self, obs):
        # reachable and not enough onions
        if obs[71] and obs[76] < NUM_ONIONS_FOR_FULL:
            return (int(obs[80]), int(obs[81]))
        elif obs[82] and obs[87] < NUM_ONIONS_FOR_FULL:
            return (int(obs[91]), int(obs[92]))
        return None
    
    def nearest_delivery(self, obs):
        return (self.pos[0] - int(obs[59]), self.pos[1] - int(obs[60]))
        
    def nearest_counter(self, obs):
        onion_pos = [(self.pos[0] - int(obs[27 + 2 * i]), self.pos[1] - int(obs[28 + 2 * i])) for i in range(4)]
        plate_pos = [(self.pos[0] - int(obs[35 + 2 * i]), self.pos[1] - int(obs[36 + 2 * i])) for i in range(4)]
        full_counters = onion_pos + plate_pos
        for i in range(4):
            counter_pos = (self.pos[0] - int(obs[63 + 2 * i]), self.pos[1] - int(obs[64 + 2 * i]))
            if counter_pos not in full_counters:
                return counter_pos
    
    def nearest_plate(self, obs):
        plate_stack_dist = sum((abs(int(obs[43])),  abs(int(obs[44]))))
        plate_dist = sum((abs(int(obs[35])),  abs(int(obs[36]))))
        if plate_stack_dist < plate_dist or plate_dist == 0: # bc no plates means (dy, dx) is (0, 0)
            return (self.pos[0] - int(obs[43]), self.pos[1] - int(obs[44]))
        else:
            return (self.pos[0] - int(obs[35]), self.pos[1] - int(obs[36]))
    
    def nearest_onion(self, obs):
        onion_stack_dist = sum((abs(int(obs[47])),  abs(int(obs[48]))))
        onion_dist = sum((abs(int(obs[27])),  abs(int(obs[28]))))
        if onion_stack_dist < onion_dist or onion_dist == 0: # bc no onions means (dy, dx) is (0, 0)
            return (self.pos[0] - int(obs[47]), self.pos[1] - int(obs[48]))
        else:
            return (self.pos[0] - int(obs[27]), self.pos[1] - int(obs[28]))
    
    def goal_tiles(self):
        if self.goal is None:
            return None
        
        r, c = self.goal
        tiles = [
            (r-1, c),  # up
            (r+1, c),  # down
            (r, c+1),  # right
            (r, c-1),  # left
        ]
        
        rows, cols = self.map.shape

        valid = []
        for r, c in tiles:
            if 0 <= r < rows and 0 <= c < cols and self.map[r, c]:
                valid.append((r, c))

        return valid
    
    def bfs(self):
        rows, cols = self.map.shape
        queue = deque([(self.pos, [])])
        visited = {self.pos}
        
        goals = self.goal_tiles()

        directions = [
            (-1,0),  # up
            (1,0),   # down
            (0,-1),   # left
            (0,1),   # right
        ]

        while queue:
            (r, c), path = queue.popleft()

            if (r, c) in goals:
                if path:
                    return path[0]
                else:
                    return 4 # if starting in goal state, return wait action

            for action, (dr, dc) in enumerate(directions):
                nr, nc = r + dr, c + dc

                if (
                    0 <= nr < rows and
                    0 <= nc < cols and
                    self.map[nr, nc] and
                    (nr, nc) not in visited
                ):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [action]))

        return None 
    
    def facing_goal(self, obs):
        if obs[0]: # right
            ahead_square = (self.pos[0], self.pos[1] + 1)
        elif obs[1]: # down
            ahead_square = (self.pos[0]  + 1, self.pos[1])
        elif obs[2]: # left
            ahead_square = (self.pos[0], self.pos[1] - 1)
        else: # up
            ahead_square = (self.pos[0] - 1, self.pos[1])
        return self.goal == ahead_square
    
    def turn_to_goal(self, obs):
        direction_tuple = (self.goal[0] - self.pos[0], self.goal[1] - self.pos[1])
        if direction_tuple == (-1, 0):
            return self.action_map['up']
        elif direction_tuple == (0, 1):
            return self.action_map['right']
        elif direction_tuple == (1, 0):
            return self.action_map['down']
        else:
            return self.action_map['left']
            
    def update_goal(self, obs):
        held_object = self.held_object(obs)
        nearest_cooked_soup = self.nearest_cooked_soup(obs)
        nearest_cooking_soup = self.nearest_cooking_soup(obs)
        nearest_full_soup = self.nearest_full_soup(obs)
        nearest_not_full_soup = self.nearest_not_full_soup(obs)
        
        # TODO: dont even use obs, just use env
        
        # have soup
        if held_object == 'soup':
            # print('try to deliver')
            self.goal = self.nearest_delivery(obs)
            self.goal_action = 'drop'
            
        # soup cooked
        elif nearest_cooked_soup is not None:
            # print('soup cooked')
            if held_object == 'plate':
                self.goal = nearest_cooked_soup
                self.goal_action = 'drop'
            elif held_object == 'onion':
                self.goal = self.nearest_counter(obs)
                self.goal_action = 'drop'
            else: # empty hand
                self.goal = self.nearest_plate(obs)
                self.goal_action = 'pickup'
                
        # pot full
        elif nearest_full_soup is not None:
            # print('full pot')
            self.goal = nearest_full_soup
            self.goal_action = 'toggle'
                
        # not full, cooking, or ready
        elif nearest_not_full_soup is not None:
            # print('no full/cooking')
            if held_object == 'plate':
                self.goal = self.nearest_counter(obs)
                self.goal_action = 'drop'
            elif held_object == 'onion':
                self.goal = nearest_not_full_soup
                self.goal_action = 'drop'
            else: # empty
                self.goal = self.nearest_onion(obs)
                self.goal_action = 'pickup'
              
        # soup cooking  
        elif nearest_cooking_soup is not None:
            # print('soup cooking')
            if held_object == 'plate':
                self.goal = nearest_cooking_soup
                self.goal_action = 'wait'
            elif held_object == 'onion':
                self.goal = self.nearest_counter(obs)
                self.goal_action = 'drop'
            else: # empty hand
                self.goal = self.nearest_plate(obs)
                self.goal_action = 'pickup'
                
        # no reachable pots
        else:
            self.goal = None
            self.goal_action = 'wait'
        
    def act(self, obs, state=None, training=True):
        obs = obs[-1] # TODO: fix for multiple agents
        self.update_position(obs)
        self.update_goal(obs)
        
        if self.goal is None:
            action = self.action_map['wait'] 
        elif self.pos in self.goal_tiles():
            if self.facing_goal(obs):
                action = self.action_map[self.goal_action]
            else:
                action = self.turn_to_goal(obs)
        else:
            move = self.bfs()
            if move is None:
                action = self.action_map['wait'] 
            else:
                action = move
        # print('in hand', self.held_object(obs))
        # print('pos:', self.pos, 'goal:', self.goal)
        # print('goal_action:', self.goal_action, 'sent action:', action)
        # print()
        # time.sleep(0.5)
        
        return torch.tensor([action]), None, None, None
        
    def update(self, next_obs):
        pass
    
    def add_to_buffer(self, obs, actions, rewards, dones, logprobs=None, values=None):
        pass
    
    def save_model(self):
        pass