import numpy as np
from snake_gameai import SnakeGameAI,Direction,Point,BLOCK_SIZE
from model import Linear_QNet,QTrainer
import random
import torch
from collections import deque
from Helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11,256,3) 
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)
        
    def get_state(self, game):
        '''
		Obtain the current status of the game
		'''
		
		# obtain head position and calculate dimensions in all 4 axes
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
		
		# we get the current address of the snake
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        '''
		Each element of the feature vector corresponds to a specific property of the game.
		The first three features represent whether there is danger in a straight line, to the right or to the left of the snake's head. It is checked whether the snake would collide with an obstacle in that direction.
		The next four features represent the snake's current direction of movement (dir_l, dir_r, dir_u, dir_d).
		The last four features represent the location of the food relative to the snake's head. Each feature indicates whether the food is to the left, right, above or below the head.
		'''
        state = [
			# Danger Straight
			(dir_u and game.is_collision(point_u))or
			(dir_d and game.is_collision(point_d))or
			(dir_l and game.is_collision(point_l))or
			(dir_r and game.is_collision(point_r)),

			# Danger right
			(dir_u and game.is_collision(point_r))or
			(dir_d and game.is_collision(point_l))or
			(dir_u and game.is_collision(point_u))or
			(dir_d and game.is_collision(point_d)),

			# Danger Left
			(dir_u and game.is_collision(point_r))or
			(dir_d and game.is_collision(point_l))or
			(dir_r and game.is_collision(point_u))or
			(dir_l and game.is_collision(point_d)),

			# Move Direction
			dir_l,
			dir_r,
			dir_u,
			dir_d,

			# Food Location
			game.food.x < game.head.x, # food is in left
			game.food.x > game.head.x, # food is in right
			game.food.y < game.head.y, # food is up
			game.food.y > game.head.y # food is down
		]
        return np.array(state, dtype=int)
    
    def get_action(self, state):
		# random moves: tradeoff explotation / exploitation
        self.epsilon = 80 - self.n_game # random prob
        final_move = [0, 0, 0]
        # if random
        if(random.randint(0, 200) < self.epsilon):
            move = random.randint(0, 2)
            final_move[move] = 1
        else: # use model
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # prediction by model
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
    
    def train_short_memory(self, state, action, reward, next_state, done):
        '''
        Train the model in the "short term"
        '''
        self.trainer.train_step(state, action, reward, next_state, done)
        
    def train_long_memory(self):
        '''
        Train the model with previous data
        '''
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
def train():
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        agent = Agent()
        game = SnakeGameAI()
        while True:
			# Get Old state
            state_old = agent.get_state(game)

			# get move
            final_move = agent.get_action(state_old)

			# perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

			# train short memory
            agent.train_short_memory(state_old,final_move,reward,state_new,done)

			#remember
            agent.remember(state_old,final_move,reward,state_new,done)
            
            if done:
				# Train long memory,plot result
                game.reset()
                agent.n_game += 1
                agent.train_long_memory()
                if(score > reward): # new High score 
                    reward = score
                    agent.model.save()
                print('Game:',agent.n_game,'Score:',score,'Record:',record)
                
                plot_scores.append(score)
                total_score+=score
                mean_score = total_score / agent.n_game
                plot_mean_scores.append(mean_score)
                plot(plot_scores,plot_mean_scores)


if(__name__=="__main__"):
    train()
