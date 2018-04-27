import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Brain:
    def __init__(self, num_features, num_actions,
                 learning_rate=0.001,
                 gamma_decay=0.99,
                 epsilon_greedy=0.9,
                 epsilon_greedy_delta=0.0002,
                 unfreeze_q=300,
                 memory_size=500,
                 hidden_units=(10, 20),
                 batch_size=32,
                 output_graph=False):
        # Initialize parameters
        self.num_actions = num_actions
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.gamma_decay = gamma_decay
        self.epsilon_greedy = 0.1
        self.epsilon_greedy_max = epsilon_greedy
        self.epsilon_greedy_delta = epsilon_greedy_delta
        self.learn_fix = (self.epsilon_greedy + self.epsilon_greedy_max) / 2
        self.unfreeze_q = unfreeze_q
        self.memory_size = memory_size
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.output_graph = output_graph

        self.update_counter = 0

        # Use a memory repository that has form [state, action, reward, state_prime]
        self.memory = np.zeros((self.memory_size, self.num_features * 2 + 2))

        # Build up neural nets
        self.build_nets()
        params_fresh = tf.get_collection("fresh_net_params")
        params_fixed = tf.get_collection("fixed_net_params")

        self.replace = [tf.assign(fixed, fresh) for fixed, fresh in zip(params_fixed, params_fresh)]

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.history = []


    def build_nets(self):
        w_initializer = tf.contrib.layers.xavier_initializer()

        # Build the first Q net (Q_fresh) which is kept fresh after every step

        # Create two placeholders for Q_fresh
        self.state_input = tf.placeholder(tf.float32, shape=(None, self.num_features), name="state_input")
        self.Q_TD = tf.placeholder(tf.float32, shape=(None, self.num_actions), name="Q_fixed")

        with tf.variable_scope("Fresh"):
            names_fresh = ["fresh_net_params", tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope("layer_1"):
                w1_fresh = tf.get_variable("w1_fresh",
                                           shape=(self.num_features, self.hidden_units[0]),
                                           initializer=w_initializer,
                                           collections=names_fresh)
                layer_1_fresh = tf.nn.relu(tf.matmul(self.state_input, w1_fresh))

            with tf.variable_scope("layer_2"):
                w2_fresh = tf.get_variable("w2_fresh",
                                           shape=(self.hidden_units[0], self.hidden_units[1]),
                                           initializer=w_initializer,
                                           collections=names_fresh)
                layer_2_fresh = tf.nn.relu(tf.matmul(layer_1_fresh, w2_fresh))

            with tf.variable_scope("layer_3"):
                w3_fresh = tf.get_variable("w3_fresh",
                                           shape=(self.hidden_units[1], self.num_actions),
                                           initializer=w_initializer,
                                           collections=names_fresh)
                self.Q_fresh = tf.matmul(layer_2_fresh, w3_fresh)

        # Compute the MSE between Q-network and Q-learning targets
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.Q_TD, self.Q_fresh))
        with tf.variable_scope("train"):
            # Create an optimizer
            self.training_opt = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        # Build the second Q net (Q_fixed) which is only updated to Q_fresh every hundreds of steps

        # Create a placeholder
        self.state_prime = tf.placeholder(tf.float32, shape=(None, self.num_features), name="state_prime")

        with tf.variable_scope("Fixed"):
            names_fixed = ["fixed_net_params", tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope("layer_1"):
                w1_fixed = tf.get_variable("w1_fixed",
                                           shape=(self.num_features, self.hidden_units[0]),
                                           initializer=w_initializer,
                                           collections=names_fixed)

                layer_1_fixed = tf.nn.relu(tf.matmul(self.state_prime, w1_fixed))

            with tf.variable_scope("layer_2"):
                w2_fixed = tf.get_variable("w2_fixed",
                                           shape=(self.hidden_units[0], self.hidden_units[1]),
                                           initializer=w_initializer,
                                           collections=names_fixed)
                layer_2_fixed = tf.nn.relu(tf.matmul(layer_1_fixed, w2_fixed))

            with tf.variable_scope("layer_3"):
                w3_fixed = tf.get_variable("w3_fixed",
                                           shape=(self.hidden_units[1], self.num_actions),
                                           initializer=w_initializer,
                                           collections=names_fixed)
                self.Q_next = tf.matmul(layer_2_fixed, w3_fixed)


    def add_memory(self, state, action, reward, state_prime):
        if not hasattr(self, "index"):
            self.index = 0

        new_piece = np.hstack((state, [action, reward], state_prime))

        # Add this new piece of memory by replacing the old one
        self.index = self.index % self.memory_size
        self.memory[self.index, :] = new_piece

        self.index += 1


    def choose_action(self, state):
        # Perform epsilon greedy to choose an action, given state
        Q_values = self.sess.run(self.Q_fresh, feed_dict={self.state_input: state[np.newaxis, :]})

        # (1 - epsilon) probability to explore uniformly
        dstb = np.ones(self.num_actions) * (1 - self.epsilon_greedy) / self.num_actions

        # epsilon probability to exploit optimal action
        exploit = np.argmax(Q_values)
        dstb[exploit] = dstb[exploit] + self.epsilon_greedy

        action = np.random.choice(self.num_actions, p=dstb)

        return action


    def learn_one_step(self):
        # Check first if need to update the fixed Q net
        if self.update_counter == self.unfreeze_q:
            self.sess.run(self.replace)
            self.update_counter = 0
            self.epsilon_greedy = (self.epsilon_greedy + self.learn_fix) / 2
            # print("Updated Q fixed!", self.epsilon_greedy, self.learn_fix)

        # Sample from memory
        sample_index = np.random.choice(self.memory_size, size=self.batch_size)

        batch = self.memory[sample_index, :]

        q_next, q_fresh = self.sess.run([self.Q_next, self.Q_fresh],
                                        feed_dict={self.state_input: batch[:, :self.num_features],
                                                   self.state_prime: batch[:, -self.num_features:]})

        # Now compute the TD term from q_next
        q_TD = np.copy(q_fresh)

        row_indices = np.arange(self.batch_size, dtype=np.int32)
        actions = batch[:, self.num_features].astype(int)
        rewards = batch[:, self.num_features + 1]

        q_TD[row_indices, actions] = rewards + self.gamma_decay * np.max(q_next, axis=1)

        # Feed the Q nets to perform one step of update
        _, cost = self.sess.run([self.training_opt, self.loss],
                                feed_dict={self.state_input: batch[:, :self.num_features],
                                           self.Q_TD: q_TD})

        self.update_counter += 1

    def decrease_explore(self):
        self.epsilon_greedy = (self.epsilon_greedy + self.epsilon_greedy_max) / 2
        self.learn_fix = (self.epsilon_greedy + self.epsilon_greedy_max) / 2

    def plot_history(self, title, filename):
        plt.plot(np.arange(len(self.history)), self.history)
        plt.title(title)
        plt.ylabel("Total steps in one episode")
        plt.xlabel("Learned episodes")
        plt.savefig(filename)
        plt.show()
