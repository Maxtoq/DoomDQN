import random
import torch
import time

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import vizdoom as vzd
import numpy as np

from torch.autograd import Variable
from skimage import transform
from collections import deque
from PIL import Image


class DQNetwork(nn.Module):

    def __init__(self):
        super(DQNetwork, self).__init__()
        """
            First Convolutional layer:
            Conv2d: Widht = [(Input width - Kernel width + 2 * Padding) / Stride] + 1
                => [(84 - 8 + 0) / 2] + 1 = 39 => 39 x 39 x 32
            BatchNormalization
        """
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=2)
        self.batchnorm1 = nn.BatchNorm2d(num_features=32)

        """
            Second Convolutional layer:
            Conv2d: Output => [(38 - 4 + 0) / 2] + 1 = 18 => 18 x 18 x 64
            BatchNormalization
        """
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)

        """
            Third Convolutional layer:
            Conv2d: Output => [(18 - 4 + 0) / 2] + 1 = 8 => 8 x 8 x 128
            BatchNormalization
        """
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)

        """
            First Fully Connected layer:
            Input size = 8 x 8 x 128 = 8192
        """
        self.fc1 = nn.Linear(in_features=8192, out_features=512)

        """
            Second Fully Connected layer = Output Layer
        """
        self.out = nn.Linear(in_features=512, out_features=3)

    def forward(self, x):
        # First Convolutional layer with ELU activation
        x = F.elu(self.batchnorm1(self.conv1(x)))

        # Second Convolutional layer with ELU activation
        x = F.elu(self.batchnorm2(self.conv2(x)))

        # Third Convolutional layer with ELU activation
        x = F.elu(self.batchnorm3(self.conv3(x)))

        # Flatten the output of the convolution
        x = x.view(-1, 8192)

        # First Fully Connected layer with ELU activation
        x = F.elu(self.fc1(x))

        # Output with the second fully connected layer
        return self.out(x)


class DoomDQNAgent():
    
    RESIZED_SIZE = (84, 84)

    def __init__(self):
        # Deep Q Learning agent
        # Deep Q Network
        self.dqn = DQNetwork().cuda()
        # Loss
        self.loss = nn.MSELoss()
        # Learning rate
        self.lr = 0.001
        # Discount factor
        self.dr = 0.9
        # Optimizer
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.lr)
        # Exploration rate
        self.eps = 0.001
        self.eps_step = 0.0001
        
        # Initialize the game environment
        # Create the environment
        self.game = vzd.DoomGame()
        # Load the config
        self.game.load_config('basic.cfg')
        # Load the scenario
        self.game.set_doom_scenario_path('basic.wad')
        self.game.init()
        self.screen_size = (self.game.get_screen_width(), self.game.get_screen_height())
        # Create possible actions [left, right, shoot]
        self.actions = [[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]]
        # Number of tics we skip for each action
        self.skip_rate = 4

        # Stack of 4 frames
        self.max_len = 4
        self.frame_deque = None
        
    def preprocess_frame(self, frame):
        # Grayscale the frame
        gray = np.mean(frame, 0)

        # Crop the frame
        if self.screen_size == (1920, 1080):
            cropped = gray[400:-300, 160:-160]
        elif self.screen_size == (320, 240):
            cropped = gray[80:-33, 30:-30]

        # Normalise pixel values
        normed = cropped / 255.0

        # Resize
        resized = transform.resize(normed, list(DoomDQNAgent.RESIZED_SIZE))
        
        return resized

    def stack_frame(self, frame):
        # If we start a new episode
        if self.frame_deque is None:
            # Stack the frame four times
            self.frame_deque = deque([frame, ] * self.max_len, maxlen=self.max_len)
        else:
            # Append the new frame to deque
            self.frame_deque.append(frame)

    def get_q_values(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        state = state.type(torch.FloatTensor).cuda()
        state = Variable(state)
        return self.dqn(state)

    def choose_action(self, state):
        # Perform Epsilon-Greedy Policy
        if random.random() > self.eps:
            # Perform Exploration
            a = random.randint(0, len(self.actions) - 1)
        else:
            # Perform Exploitation
            q_values = self.get_q_values(state)
            a = int(torch.argmax(q_values))
        self.eps += self.eps_step

        return self.actions[a]

    def learning_step(self, state, action, reward, new_state):
        """ Compute Q-target and update parameters. """
        if new_state is not None:
            # Q-target = Reward + discount * maxQ(new_state)
            # Get maximum Q-value for the next state
            q2 = np.max(self.get_q_values(new_state).data.cpu().numpy(), axis=1)
            # Compute Q-values for the current state (in case we performed exploration)
            q_target = self.get_q_values(state).data.cpu().numpy()
            # Update to obtain Q-target
            q_target[0, action] = reward + self.dr * q2
        else:
            # Compute Q-values for the current state (in case we performed exploration)
            q_target = self.get_q_values(state).data.cpu().numpy()
            # Update to obtain Q-target
            q_target[0, action] = reward
        
        # Execute Learning step
        q_target = Variable(torch.from_numpy(q_target).cuda())
        output = self.get_q_values(state)
        loss = self.loss(output, q_target)
        # Reinitialize gradients
        self.optimizer.zero_grad()
        # Compute gradients and update parameters
        loss.backward()
        self.optimizer.step()

    def run_game(self, nb_episode=1):
        for i in range(nb_episode):
            # Start new episode
            self.game.new_episode()
            
            while not self.game.is_episode_finished():
                game_state = self.game.get_state()
                
                # Get image and preprocess it
                frame = game_state.screen_buffer
                preproc_frame = self.preprocess_frame(frame)

                # Stack the preprocessed frame
                self.stack_frame(preproc_frame)
                state = np.array(self.frame_deque)
                
                # Choose an action
                action = self.choose_action(state)
                print(action)
                
                # Make action and get reward
                reward = self.game.make_action(action, self.skip_rate)
                print ("\treward:", reward)

                # Get new state
                if not self.game.is_episode_finished():
                    new_state_frame = self.preprocess_frame(self.game.get_state().screen_buffer)
                    self.stack_frame(new_state_frame)
                    new_state = np.array(self.frame_deque)
                else:
                    new_state = None
                
                # Perform learning step
                self.learning_step(state, action, reward, new_state)

            print ("Result:", self.game.get_total_reward())
            
        self.game.close()


if __name__ == "__main__":
    d = DoomDQNAgent()
    d.run_game()
