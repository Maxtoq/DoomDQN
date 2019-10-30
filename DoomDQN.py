import random
import time

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import vizdoom as vzd
import numpy as np

from skimage import transform
from collections import deque
from PIL import Image


class DQNetwork(nn.Module):

    def __init__(self):
        super(DQNetwork, self).__init__()
        """
            First Convolutional layer:
            Conv2d: Output size = [(Input size - Kernel size + 2 * Padding) / Stride] + 1
                => 
            BatchNormalization
        """
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=2)
        self.batchnorm1 = nn.BatchNorm2d(num_features=32)

        """
            Second Convolutional layer:
            Conv2d
            BatchNormalization
        """
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)

        """
            Third Convolutional layer:
            Conv2d
            BatchNormalization
        """
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)

        """
            First Fully Connected layer:

        """

    def forward(self, x):
        # First Convolutional layer with ELU activation
        x = F.elu(self.batchnorm1(self.conv1(x)))

        # Second Convolutional layer with ELU activation
        x = F.elu(self.batchnorm2(self.conv2(x)))

        # Third Convolutional layer with ELU activation
        x = F.elu(self.batchnorm3(self.conv3(x)))


class DQLAgent():

    def __init__(self):
        self.dqn = DQNetwork()        

    def choose_action(self, frames):
        return [0, 0, 1]


class DoomEnv():
    
    RESIZED_SIZE = (84, 84)

    def __init__(self):
        # Create the environment
        self.game = vzd.DoomGame()

        # DQN
        self.dql_agent = DQLAgent()
        
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
        resized = transform.resize(normed, list(DoomEnv.RESIZED_SIZE))
        
        return resized

    def stack_frame(self, frame):
        # If we start a new episode
        if self.frame_deque is None:
            # Stack the frame four times
            self.frame_deque = deque([frame, ] * self.max_len, maxlen=self.max_len)
        else:
            # Append the new frame to deque
            self.frame_deque.append(frame)
    
    def run_game(self, nb_episode=1):
        for i in range(nb_episode):
            # Start new episode
            self.game.new_episode()
            
            while not self.game.is_episode_finished():
                state = self.game.get_state()
                
                # Get image and preprocess it
                frame = state.screen_buffer
                preproc_frame = self.preprocess_frame(frame)

                # Stack the preprocessed frame
                self.stack_frame(preproc_frame)
                
                # Choose an action
                #action = random.choice(self.actions)
                action = self.dql_agent.choose_action(np.array(self.frame_deque))
                print(action)
                
                reward = self.game.make_action(action, self.skip_rate)
                print ("\treward:", reward)
                
                time.sleep(0.1)
            print ("Result:", self.game.get_total_reward())
            
        self.game.close()


if __name__ == "__main__":
    d = DoomEnv()
    d.run_game()
