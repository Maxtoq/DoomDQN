import random
import time

import matplotlib.pyplot as plt
import numpy as np

from skimage import transform
from vizdoom import DoomGame
from PIL import Image


class DQNAgent():

    def __init__(self):
        self.a = 1


class DoomEnv():
    
    def __init__(self):
        # Create the environment
        self.game = DoomGame()
        
        # Load the config
        self.game.load_config('basic.cfg')
        
        # Load the scenario
        self.game.set_doom_scenario_path('basic.wad')
        self.game.init()
        
        # Create possible actions [left, right, shoot]
        self.actions = [[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]]
        
        # Number of tics we skip for each action
        self.skip_rate = 4

        # Stack of 4 frames
        
        
    def preprocess_frame(self, frame):
        # Grayscale the frame
        gray = np.mean(frame, 0)

        # Crop the frame
        cropped = gray[80:-33, 30:-30]

        # Normalise pixel values
        normed = cropped / 255.0

        # Resize
        resized = transform.resize(normed, [84, 84])
        
        return resized

    def stack_frame(self, frame):

    
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
                
                action = random.choice(self.actions)
                print(action)
                
                reward = self.game.make_action(action, self.skip_rate)
                print ("\treward:", reward)
                
                time.sleep(0.1)
            print ("Result:", self.game.get_total_reward())
            time.sleep(2)
            
        self.game.close()


if __name__ == "__main__":
    d = DoomEnv()
    d.run_game()
