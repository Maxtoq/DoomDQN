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


def plot_list(l, legend):
    plt.plot(np.arange(len(l)), l, label=legend)
    plt.legend()

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


class ReplayMemory():

    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)

    def store(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)

        if buffer_size < batch_size:
            batch_size = buffer_size
            print(f'Not enough experience stored, returning only {batch_size} experiences.')

        # Get random ids in the buffer
        ids = np.random.choice(buffer_size, size=batch_size, replace=False)

        return [self.buffer[i] for i in ids]


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

        # Replay Memory
        self.memory = ReplayMemory()
        # Batch size when learning
        self.batch_size = 64

        # Stack of 4 frames
        self.max_len = 4
        self.frame_deque = None

        # Historic of metrics
        self.hist_loss = []
        self.hist_reward = []

    def display_metrics(self):
        plot_list(self.hist_loss, 'Loss')
        plot_list(self.hist_reward, 'Reward')
        plt.show()        
        
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
        state = torch.from_numpy(state)
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        state = state.type(torch.FloatTensor).cuda()
        state = Variable(state)
        return self.dqn(state)

    def choose_action(self, state, nb_epoch=None, i=None):
        # Compute Epsilon
        if nb_epoch is None:
            eps = 1
        else:
            eps = (i / nb_epoch) * 0.9

        # Perform Epsilon-Greedy Policy
        if random.random() > eps:
            # Perform Exploration
            a = random.randint(0, len(self.actions) - 1)
        else:
            # Perform Exploitation
            q_values = self.get_q_values(state)
            a = int(torch.argmax(q_values))

        return self.actions[a]

    def learning_step(self):
        """ Perform a learning step using a batch from the replay memory. """
        batch = self.memory.sample(self.batch_size)

        states_b = np.array([each[0] for each in batch], ndmin=3)
        actions_b = np.array([each[1] for each in batch])
        rewards_b = np.array([each[2] for each in batch]) 
        new_states_b = np.array([each[3] for each in batch], ndmin=3)

        # Get Q-values of next states
        next_qs_b = self.get_q_values(new_states_b).data.cpu().numpy()

        # Initialize Q-targets as Q-values for the current state
        q_targets_b = self.get_q_values(states_b).data.cpu().numpy()

        for i in range(len(batch)):
            if new_states_b[i] is not None:
                # Q-target = Reward + discount * maxQ(new_state)
                # Get maximum Q-value for the next state
                q2 = np.max(next_qs_b[i])
                # Update to obtain Q-target
                q_targets_b[i, action] = rewards_b[i] + self.dr * q2
            else:
                # Update to obtain Q-target
                q_targets_b[i, action] = rewards_b[i]
            
        # Execute Learning step
        q_targets_b = Variable(torch.from_numpy(q_targets_b).cuda())
        outputs_b = self.get_q_values(states_b)
        loss = self.loss(outputs_b, q_targets_b)
        # Reinitialize gradients
        self.optimizer.zero_grad()
        # Compute gradients and update parameters
        loss.backward()
        self.optimizer.step()

        return float(torch.mean(loss))

    def run_train(self, nb_epoch=100, nb_episodes=20):
        for i in range(nb_epoch):
            # Gather experiences playing
            for j in range(nb_episodes):
                # Start new episode
                self.game.new_episode()

                # Create new hist loss for this episode
                loss = 0
                
                while not self.game.is_episode_finished():
                    game_state = self.game.get_state()
                    
                    # Get image and preprocess it
                    frame = game_state.screen_buffer
                    preproc_frame = self.preprocess_frame(frame)

                    # Stack the preprocessed frame
                    self.stack_frame(preproc_frame)
                    state = np.array(self.frame_deque)
                    
                    # Choose an action
                    action = self.choose_action(state, nb_epoch, i + 1)
                    
                    # Make action and get reward
                    reward = self.game.make_action(action, self.skip_rate)

                    # Get new state
                    if not self.game.is_episode_finished():
                        new_state_frame = self.preprocess_frame(self.game.get_state().screen_buffer)
                        self.stack_frame(new_state_frame)
                        new_state = np.array(self.frame_deque)
                    else:
                        new_state = None
                    
                    # Store experience in Replay memory
                    self.replay_memory.store((state, action, reward, new_state))
            #     # Perform learning step
            #     loss = self.learning_step(state, action, reward, new_state)

            # # Add hist data
            # self.hist_loss.append(loss)
            # self.hist_reward.append(self.game.get_total_reward())

            # Perform learning step
            self.learning_step()

            print (f"Ep #{i} Result:", self.hist_reward[-1])
            
        self.game.close()

        self.display_metrics()


if __name__ == "__main__":
    d = DoomDQNAgent()
    d.run_train()
