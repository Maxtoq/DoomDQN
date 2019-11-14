# DoomDQN
In this repositery I build a Deep Q-Network to play Doom, following the tutorial: https://www.freecodecamp.org/news/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8/.

## Task
The first goal is to play a very simple scenario of the game: a monster is in the room; we can go right, left or shoot; we need to navigate in order to put the monster in front of us and shoot.

## Architecture
In order to capture movement, we stack four frames of the game to construct our state.

We use a **CNN** to analyse the frames. It's composed of three convolutional layers each followed by Batch Normalisation, one fully connected layer. A final output layer gives us the predicted Q-values for the three possible actions.

In order to reduce correlation between experiences, we use **experience replay**. We save the experiences tuple (state, action, reward, new_state) in a replay memory. After each episode, we train our DQN on a batch of experiences.

## Next steps
After reaching good results on the first task, I'll continue the development of this project on more difficult tasks of the Doom environment. Following the Thomas Simonini's tutorials, I'll be implementing Dueling Double DQN, Prioritized Experience Replay, Policy Gradients, A2C and maybe more ! The goal is to create an agent that could complete the entire game of Doom... let's get to work !
