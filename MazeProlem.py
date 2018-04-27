import numpy as np
import matplotlib.pyplot as plt
from maze import *
from evaluation import *
from value_plot import *


def Q_value_iteration(environment, thresh, num_iter, gamma=0.9):
    num_state = environment.snum
    num_action = environment.anum

    Q_curr = np.zeros((num_state, num_action))
    Q_next = np.zeros((num_state, num_action))

    # Start value iteration
    for iter in range(num_iter):
        # Iterate over all states
        for state in range(num_state):
            # Ignore the Q pairs for goal states
            if (state > 95 and state < 104):
                continue

            for action in range(num_action):
                # Add the true action part
                reward, next_state, done = environment.step(state, action, MDP=True)
                Q_next[state, action] = 0.9 * (reward + gamma * np.max(Q_curr[next_state, :]))

                # Add the slippage part
                slip_action = find_slip(action)
                reward, next_state, _ = environment.step(state, slip_action, MDP=True)
                Q_next[state, action] = Q_next[state, action] + 0.1 * (reward + gamma * np.max(Q_curr[next_state, :]))

        # Check convergence
        Q_delta = np.linalg.norm(Q_next - Q_curr)
        print("Iter:", iter, " Q_delta:", Q_delta)

        if Q_delta < thresh:
            break

        Q_curr = np.copy(Q_next)

    policy = np.argmax(Q_next, axis=1)

    return Q_next, policy


def Q_learning(environment, Q_true, learning_rate=0.25, epsilon=0.25, gamma=0.9, num_iter=5000, selection="epsilon-greedy"):
    num_state = environment.snum
    num_action = environment.anum

    Q = np.zeros((num_state, num_action))

    delta = []

    for iter in range(num_iter):
        # Restart at the start state
        state = environment.reset()
        done = False

        while not done:
            # Sample the action
            action = choose_action(Q[state, :], epsilon, selection, iter)

            # Transition stochastically and collect reward
            reward, next_state, done = environment.step(state, action, MDP=False)

            # Update one step of Q
            Q[state, action] = (1 - learning_rate) * Q[state, action] + \
                               learning_rate * (reward + gamma * np.max(Q[next_state, :]))

            state = next_state

        delta.append(np.linalg.norm(Q_true - Q))

        if iter % 50 == 0:
            print("Iter:", iter)

    policy = np.argmax(Q, axis=1)

    plt.plot(np.arange(len(delta)), delta)
    plt.title("RMSE of Q and Q*")
    plt.ylabel("RMSE")
    plt.xlabel("Episodes")
    plt.savefig("RMSE_Q")
    plt.show()

    return Q, policy


def choose_action(Q_values, epsilon, selection, iter):
    num_action = Q_values.shape[0]

    if selection == "epsilon-greedy":
        # (1 - epsilon) probability to explore uniformly
        dstb = np.ones(num_action) * (1 - epsilon) / num_action

        # epsilon probability to exploit optimal action
        exploit = np.argmax(Q_values)
        dstb[exploit] = dstb[exploit] + epsilon

    if selection == "softmax":
        # Implant exploitation & exploration into softmax probability
        dstb = np.exp(Q_values)
        dstb = dstb / np.sum(Q_values)

    action = np.random.choice(num_action, p=dstb)

    return action


if __name__ == "__main__":
    print("Welcome to the Maze problem!")
    env = Maze()

    choice = int(input("Enter '1' to run MDP value iteration, or '2' to run Q-learning\n"))
    if choice == 1:
        Q, policy = Q_value_iteration(env, thresh=0.00001, num_iter=200)
        np.save("Q_values", Q)
    if choice == 2:
        Q_true = np.load("Q_values.npy")
        Q, policy = Q_learning(env, Q_true)
        value_plot(Q, env)


    total_step, total_reward = evaluation(env, Q)
