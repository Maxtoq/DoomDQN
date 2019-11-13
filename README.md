# DoomDQN
In this repositery I build a Deep Q-Network to play Doom, following the tutorial: https://www.freecodecamp.org/news/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8/.

## Task
The first goal is to play a very simple scenario of the game: a monster is in the room; we can go right, left or shoot; we need to navigate in order to put the monster in front of us and shoot.

## Architecture
In order to capture movement, we stack four frames of the game to construct our state.

We use a **CNN** to analyse the frames. It's composed of three convolutional layers each followed by Batch Normalisation, one fully connected layer. A final output layer gives us the predicted Q-values for the three possible actions.

In order to reduce correlation between experiences, we use **experience replay**. aefonzeg
