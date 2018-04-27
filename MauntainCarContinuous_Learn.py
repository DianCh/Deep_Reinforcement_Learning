import numpy as np
import ReinforceContinuous as RFC
import gym


def main_ReinforceContinuous():
    env = gym.make("MountainCarContinuous-v0")
    env = env.unwrapped

    brain = RFC.Brain(num_features=2,
                      learning_rate=0.002,
                      hidden_units=(10, 20),
                      batch_size=512,
                      num_epochs=1)

    for i in range(5000):
        state = env.reset()
        done = False

        while not done:

            action = brain.choose_action(state)

            state_next, reward, done, info = env.step(action)

            brain.add_memory(state, action, reward)

            state = state_next

        # After one episode is done, learn from this episode
        brain.learn_from_episode()

    brain.plot_history("MountainCarContinuous-v0 Reinforce Learning Curve", filename="MoutainCarContinuous")


if __name__ == "__main__":
    print("Welcome to MountainCarContinuous-v0!")
    main_ReinforceContinuous()
