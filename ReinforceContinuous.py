import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Brain:
    def __init__(self, num_features,
                 learning_rate=0.01,
                 gamma_decay=0.99,
                 hidden_units=(10, 20),
                 batch_size=256,
                 num_epochs=10,
                 output_graph=False):

        self.num_features = num_features
        self.gamma_decay = gamma_decay
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.states = np.zeros((0, self.num_features))
        self.actions = np.zeros(0,)
        self.rewards = np.zeros(0,)
        self.vt = np.zeros(0,)

        self.history = []

        self.build_net()

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())


    def build_net(self):
        initializer = tf.contrib.layers.xavier_initializer()

        with tf.name_scope("inputs"):
            self.states_input = tf.placeholder(tf.float32, shape=(None, self.num_features), name="states")
            self.actions_input = tf.placeholder(tf.float32, shape=(None, ), name="actions")
            self.values_input = tf.placeholder(tf.float32, shape=(None, ), name="rewards")

        with tf.name_scope("Pi-net"):

            with tf.variable_scope("layer_1"):
                w1 = tf.get_variable("w1",
                                     shape=(self.num_features, self.hidden_units[0]),
                                     initializer=initializer)
                layer_1 = tf.nn.relu(tf.matmul(self.states_input, w1))

            with tf.variable_scope("layer_2"):
                w2 = tf.get_variable("w2",
                                     shape=(self.hidden_units[0], self.hidden_units[1]),
                                     initializer=initializer)
                layer_2 = tf.nn.relu(tf.matmul(layer_1, w2))

            with tf.variable_scope("layer_3"):
                w3 = tf.get_variable("w3",
                                     shape=(self.hidden_units[1], self.num_features),
                                     initializer=initializer)
                prob_params = tf.matmul(layer_2, w3)

            self.sigma = 0.3
            self.mu = tf.reduce_sum(tf.multiply(self.states_input, prob_params), axis=1)
            self.action_prob = tf.exp(- 0.5 * tf.square((self.actions_input - self.mu) / self.sigma)) / (np.sqrt(2 * np.pi) * self.sigma)

        with tf.name_scope("loss"):
            # The quantity log(probs)*vt is the target value to maximize, so we view its negative as the loss
            # in common DL problems
            neg_log_probs = - tf.log(self.action_prob)
            self.loss = tf.reduce_mean(tf.multiply(neg_log_probs, self.values_input))

        with tf.name_scope("train"):
            self.training_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def add_memory(self, state, action, reward):
        self.states = np.concatenate((self.states, state[np.newaxis, :]), axis=0)
        self.actions = np.append(self.actions, action)
        self.rewards = np.append(self.rewards, reward)


    def process_memory(self):
        memory_length = self.rewards.shape[0]

        # Compute the running discounted cumulative rewards
        rewards_cumulative = np.zeros(memory_length,)

        # Loop over the entire memory backwards
        reward_sum = 0
        for t in reversed(range(memory_length)):
            # Compute the cumulated rewards
            reward_sum = reward_sum * self.gamma_decay + self.rewards[t]
            rewards_cumulative[t] = reward_sum

        # Normalize the cumulative rewards
        rewards_cumulative -= np.mean(rewards_cumulative)
        # rewards_cumulative /= np.std(rewards_cumulative)

        return rewards_cumulative


    def choose_action(self, state):
        # Pass the current state to calculate the pi probabilities
        pi_mu = self.sess.run(self.mu, feed_dict={self.states_input: state[np.newaxis, :]})

        # Use the probabilities to sample a continuous action
        action = np.random.normal(loc=pi_mu, scale=self.sigma)

        return action


    def learn_from_episode(self):
        self.vt = self.process_memory()

        memory_length = self.vt.shape[0]
        self.history.append(memory_length)

        print(self.vt.shape)
        num_batches = int(memory_length / self.batch_size)

        for k in range(self.num_epochs):
            # Shuffle data in each epoch
            shuffle = np.arange(memory_length)
            np.random.shuffle(shuffle)

            for i in range(num_batches):
                # Gather a mini batch of data
                batch_indices = shuffle[i * self.batch_size:(i + 1) * self.batch_size]
                states_batch = self.states[batch_indices, :]
                actions_batch = self.actions[batch_indices]
                vt_batch = self.vt[batch_indices]

                # Run one batch
                _, loss = self.sess.run([self.training_opt, self.loss],
                                        feed_dict={self.states_input: states_batch,
                                                   self.actions_input: actions_batch,
                                                   self.values_input: vt_batch})

            # The remainder mini batch
            batch_indices = shuffle[num_batches * self.batch_size:]
            states_batch = self.states[batch_indices, :]
            actions_batch = np.array(self.actions)[batch_indices]
            vt_batch = self.vt[batch_indices]

            _, loss = self.sess.run([self.training_opt, self.loss],
                                    feed_dict={self.states_input: states_batch,
                                               self.actions_input: actions_batch,
                                               self.values_input: vt_batch})

        # Clear the memory
        self.states = np.zeros((0, self.num_features))
        self.actions = np.zeros(0,)
        self.rewards = np.zeros(0, )
        self.vt = np.zeros(0, )


    def plot_history(self, title, filename):
        plt.plot(np.arange(len(self.history)), self.history)
        plt.title(title)
        plt.ylabel("Total steps in one episode")
        plt.xlabel("Learned episodes")
        plt.savefig(filename)
        plt.show()