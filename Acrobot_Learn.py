import numpy as np
import DeepQNetwork as DQN
import Reinforce as RF
import gym


def main_DQN():
    env = gym.make("Acrobot-v1")
    env = env.unwrapped

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    brain = DQN.Brain(num_features=6,
                      num_actions=3,
                      learning_rate=0.001,
                      unfreeze_q=300,
                      memory_size=3000,
                      hidden_units=(10, 20))

    total_steps = 0

    for i in range(100):
        state = env.reset()
        done = False

        step_counter = 0

        while not done:

            action = brain.choose_action(state)

            state_next, reward, done, info = env.step(action)

            # Compute an intuitive reward for every step
            cos_theta1, sin_theta1, cos_theta2, sin_theta2, thetaDot1, thetaDot2 = state_next
            theta1 = np.arctan2(sin_theta1, cos_theta1)
            theta2 = np.arctan2(sin_theta2, cos_theta2)
            reward = - (cos_theta1 + np.cos(theta1 + theta2))

            brain.add_memory(state, action, reward, state_next)

            # Start learning once the memory has been filled up
            if total_steps > brain.memory_size:
                brain.learn_one_step()

            state = state_next

            total_steps += 1
            step_counter += 1

            if step_counter > 5000:
                print("Exceeded!")
                break

        if done:
            brain.decrease_explore()
            brain.history.append(step_counter)
            print("Episode:", i, " Length:", step_counter)

    brain.plot_history("Acrobot-v1 DQN Learning Curve", filename="Acrobot-DQN")


def main_Reinforce():
    env = gym.make("Acrobot-v1")
    env = env.unwrapped

    brain = RF.Brain(num_features=6,
                     num_actions=3,
                     learning_rate=0.001,
                     hidden_units=(10, 20),
                     batch_size=512,
                     num_epochs=3)

    for i in range(2000):
        state = env.reset()
        done = False

        step_counter = 0

        while not done:

            action = brain.choose_action(state)

            state_next, reward, done, info = env.step(action)

            brain.add_memory(state, action, reward)

            state = state_next

            # Check if this episode is too long; if so, terminate and start a new episode
            step_counter += 1
            if step_counter > 5000:
                print("Exceeded!")
                brain.clear_memory()
                break

        if done:
            # After one episode is done, learn from this episode
            brain.learn_from_episode(i)

    brain.plot_history("Acrobot-v1 REINFORCE Learning Curve", filename="Acrobot-REINFORCE")


if __name__ == "__main__":
    choice = int(input("Welcome to Acrobot-v1! Enter '1' for DQN or '2' for REINFORCE\n"))

    if choice == 1:
        main_DQN()
    if choice == 2:
        main_Reinforce()