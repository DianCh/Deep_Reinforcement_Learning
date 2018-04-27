import numpy as np
import DeepQNetwork as DQN
import Reinforce as RF
import gym


def main_DQN():
    env = gym.make("MountainCar-v0")
    env = env.unwrapped

    brain = DQN.Brain(num_features=2,
                      num_actions=3,
                      learning_rate=0.001,
                      unfreeze_q=300,
                      memory_size=4000)

    total_steps = 0

    for i in range(100):
        state = env.reset()
        done = False

        step_counter = 0

        while not done:

            action = brain.choose_action(state)

            state_next, reward, done, info = env.step(action)

            # # Compute an intuitive reward for every step
            position, velocity = state_next
            reward = np.abs(position - (-0.5))

            brain.add_memory(state, action, reward, state_next)

            # Start learning once the memory has been filled up
            if total_steps > brain.memory_size:
                brain.learn_one_step()

            state = state_next

            total_steps += 1
            step_counter += 1
            if step_counter > 10000:
                print("Exceeded!")
                break

        if done:
            brain.decrease_explore()
            brain.history.append(step_counter)
            print("Episode:", i, " Length:", step_counter)

    brain.plot_history("MountainCar-v0 DQN Learning Curve", filename="MoutainCar-DQN")


def main_Reinforce():
    env = gym.make("MountainCar-v0")
    env = env.unwrapped

    brain = RF.Brain(num_features=2,
                     num_actions=3,
                     learning_rate=0.001,
                     hidden_units=(10, 20),
                     batch_size=512,
                     num_epochs=3)

    for i in range(2000):
        state = env.reset()
        done = False

        while not done:

            action = brain.choose_action(state)

            state_next, reward, done, info = env.step(action)

            brain.add_memory(state, action, reward)

            state = state_next

        # After one episode is done, learn from this episode
        brain.learn_from_episode(i)

    brain.plot_history("MountainCar-v0 REINFORCE Learning Curve", filename="MoutainCar-REINFORCE")


if __name__ == "__main__":
    choice = int(input("Welcome to MountainCar-v0! Enter '1' for DQN or '2' for REINFORCE\n"))

    if choice == 1:
        main_DQN()
    if choice == 2:
        main_Reinforce()