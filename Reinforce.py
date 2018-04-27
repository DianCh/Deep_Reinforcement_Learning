import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Brain:
    def __init__(self, num_features, num_actions,
                 learning_rate=0.01,
                 gamma_decay=0.99,
                 hidden_units=(10, 20),
                 batch_size=256,
                 num_epochs=10,
                 output_graph=False):

        self.num_features = num_features
        self.num_actions = num_actions
        self.gamma_decay = gamma_decay
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.states = np.zeros((0, self.num_features))
        self.actions = []
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
            self.actions_input = tf.placeholder(tf.int32, shape=(None, ), name="actions")
            self.values_input = tf.placeholder(tf.float32, shape=(None, ), name="rewards")

        with tf.name_scope("Pi-net"):
            layer_1 = tf.layers.dense(inputs=self.states_input,
                                      units=self.hidden_units[0],
                                      activation=tf.nn.relu,
                                      kernel_initializer=initializer,
                                      bias_initializer=tf.constant_initializer(0.1),
                                      name="fc1")

            layer_2 = tf.layers.dense(inputs=layer_1,
                                      units=self.hidden_units[1],
                                      activation=tf.nn.relu,
                                      kernel_initializer=initializer,
                                      name="fc2")

            action_score = tf.layers.dense(inputs=layer_2,
                                           units=self.num_actions,
                                           activation=None,
                                           kernel_initializer=initializer,
                                           bias_initializer=tf.constant_initializer(0.1),
                                           name="action-score")

            self.action_probs = tf.nn.softmax(action_score, name="action-probs")

        with tf.name_scope("loss"):
            # The quantity log(probs)*vt is the target value to maximize, so we view its negative as the loss
            # in common DL problems
            neg_log_probs = tf.reduce_sum(- tf.log(self.action_probs) *
                                            tf.one_hot(indices=self.actions_input, depth=self.num_actions),
                                            axis=1)
            self.loss = tf.reduce_mean(neg_log_probs * self.values_input)

        with tf.name_scope("train"):
            self.training_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def add_memory(self, state, action, reward):
        self.states = np.concatenate((self.states, state[np.newaxis, :]), axis=0)
        self.actions.append(action)
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
        rewards_cumulative /= np.std(rewards_cumulative)

        return rewards_cumulative


    def choose_action(self, state):
        # Pass the current state to calculate the pi probabilities
        pi_probs = self.sess.run(self.action_probs, feed_dict={self.states_input: state[np.newaxis, :]})

        # Use the probabilities to sample an action
        action = np.random.choice(self.num_actions, p=pi_probs.ravel())

        return action


    def learn_from_episode(self, episode):
        self.vt = self.process_memory()
        memory_length = self.vt.shape[0]
        print("Episode:", episode, " Length:", memory_length)

        self.history.append(memory_length)

        num_batches = int(memory_length / self.batch_size)

        for k in range(self.num_epochs):
            # Shuffle data in each epoch
            shuffle = np.arange(memory_length)
            np.random.shuffle(shuffle)

            for i in range(num_batches):
                # Gather a mini batch of data
                batch_indices = shuffle[i * self.batch_size:(i + 1) * self.batch_size]
                states_batch = self.states[batch_indices, :]
                actions_batch = np.array(self.actions)[batch_indices]
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
        self.actions = []
        self.rewards = np.zeros(0, )
        self.vt = np.zeros(0, )


    def clear_memory(self):
        self.states = np.zeros((0, self.num_features))
        self.actions = []
        self.rewards = np.zeros(0, )
        self.vt = np.zeros(0, )


    def plot_history(self, title, filename):
        plt.plot(np.arange(len(self.history)), self.history)
        plt.title(title)
        plt.ylabel("Total steps in one episode")
        plt.xlabel("Learned episodes")
        plt.savefig(filename)
        plt.show()
