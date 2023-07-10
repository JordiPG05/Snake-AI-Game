import torch.nn as nn
import os
import torch
import torch.nn.functional as F
import  torch.optim as optim
class Linear_QNet(nn.Module):
    '''
    We create the basis of the neural network
    '''
    def __init__(self, input_size, hidden_size, output_size):
         '''
         Input sizes, dense layer neurons and output sizes are received.
        '''
         super().__init__()
         self.linear1 = nn.Linear(input_size, hidden_size) # we define the input up to the dense layer
         self.linear2 = nn.Linear(hidden_size, output_size) # we define the dense layer up to the exit
         
    def forward(self, x):
        ''' 
        We define activation functions
        '''
        x = F.relu(self.linear1(x)) # add a relu layer to linear1
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = '.\SnakeAI'
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)

class QTrainer:
    '''
    This class will train the neural network
    '''
    def __init__(self,model,lr,gamma):
        # Learning Rate for Optimizer
        self.lr = lr
        
        # Discount Rate
        self.gamma = gamma
        
        # Linear NN defined above.
        self.model = model
        
        # optimizer for weight and biases updation
        self.optimer = optim.Adam(model.parameters(),lr = self.lr)
        
        # Mean Squared error loss function
        self.criterion = nn.MSELoss()
        
    def train_step(self,state,action,reward,next_state,done):
        '''
        We apply the formula for the modified Bellman equation
        '''
        # transform input data to tensors, and define dtypes
        state = torch.tensor(state,dtype=torch.float)  # original state
        next_state = torch.tensor(next_state,dtype=torch.float) # post state
        action = torch.tensor(action,dtype=torch.long) # action
        reward = torch.tensor(reward,dtype=torch.float) # reward
        
        # ensure that subsequent calculations are consistent (bidimensional data)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)
            
        # calculate Q-value for current state
        pred = self.model(state) # predict with model
        target = pred.clone()
        
        # calculate Q_new
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]: # if done is not False (game finished)
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])) # calculate Q_new (next step)
            target[idx][torch.argmax(action).item()] = Q_new # we define target taking into account the following step
        
        # Model feedback and tuning
        self.optimer.zero_grad() # reset gradients (remove accumulation)
        loss = self.criterion(target, pred) # calculate loss function between target and pred
        loss.backward() # search for optimum weights with backpropagation
        
        self.optimer.step() # update model weights
            


