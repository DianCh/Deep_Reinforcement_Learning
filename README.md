# Deep Reinforcement Learning
This repository contains implementations of several Deep-Learning-based reinforcement learning algorithms, with experiments on **OpenAI gym** models. Current algorithms include:

- Q-value iteration (toy example, non-deep-learning)
- DQN
- REINFORCE for discrete actions
- REINFORCE for continuous actions

The collection of algorithms will continue to grow.

----

## Run tests
To run the algorithms, simply execute their corresponding main script:

- MazeProblem.py
- Acrobot_Learn.py
- MountainCar_Learn.py
- MountainCarContinous_Learn.py

You will see prompts with instructions that let you choose the RL algorithm to use.

Feel free to play around with the hyper-parameters.

**Tips:** **Turning off the animation** boosts the speed of execution. You can switch on the animation by modifying the main scripts (you may see the agents do stupid things for a long time if the learning hasn't convergenged).

## Interface & Algorithms
Most of the algorithms are implemented with independence on the actual environment & agent, just so to achieve a "plug and play" style. Environments or models can be connected to different "brains". Here the brains are:

- DeepQNetwork.py			(DQN)
- Reinforce.py				(REINFORCE discrete)
- ReinforceContinous.py		(REINFORCE continous)

## Dependencies
For you to run the tests, python 3 with normal scientific packages (numpy, matplotlib, etc.) would suffice. In addition, you need:

1. TensorFlow (1.5.0 guaranteed to work, no gpu required)
[Instructions for installation](https://www.tensorflow.org/install/)
2. OpenAI gym (0.10.4 guaranteed to work)
[Instructions for installation](https://gym.openai.com/docs/)

![alt text](https://github.com/DianCh/Deep_Reinforcement_Learing/blob/master/results/moutaincar.png "MountainCar in gym")

## Sample Results
Here are some of the learning curves of the running algorithms:
![alt text](https://github.com/DianCh/Deep_Reinforcement_Learing/blob/master/results/MountainCar-DQN.png "MountainCar DQN")

![alt text](https://github.com/DianCh/Deep_Reinforcement_Learing/blob/master/results/Acrobot-DQN.png "Acrobot DQN")

![alt text](https://github.com/DianCh/Deep_Reinforcement_Learing/blob/master/results/MountainCar-REINFORCE.png "MountainCar REINFORCE")
