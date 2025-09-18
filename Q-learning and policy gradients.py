import numpy as np
import tensorflow as tf

# -------------------------------
# Environment setup
# -------------------------------
grid_size = 5
n_states = grid_size * grid_size
n_actions = 4  # up, down, left, right

# Reward matrix
rewards = np.full((n_states,), -1)  # -1 for all states
rewards[24] = 10   # Goal state
rewards[12] = -10  # Pitfall state

# -------------------------------
# Q-learning parameters
# -------------------------------
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Initialize Q-table
Q_table = np.zeros((n_states, n_actions))

# Epsilon-greedy action selection
def epsilon_greedy_action(Q_table, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, n_actions)  # Explore
    else:
        return np.argmax(Q_table[state])  # Exploit

# -------------------------------
# Q-learning loop
# -------------------------------
for episode in range(1000):
    state = np.random.randint(0, n_states)
    done = False
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        next_state = np.random.randint(0, n_states)  # Random next state for simulation
        reward = rewards[next_state]

        # Bellman update
        Q_table[state, action] += alpha * (reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action])

        state = next_state
        if next_state in {24, 12}:
            done = True

print("Q-learning training completed.\n")

# -------------------------------
# Policy Gradient (REINFORCE) setup
# -------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(n_states,)),
    tf.keras.layers.Dense(n_actions, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Choose action based on policy network
def get_action(state):
    state_input = tf.one_hot(state, n_states)
    state_input = tf.expand_dims(state_input, axis=0)  # Add batch dimension
    action_probs = model(state_input)
    return np.random.choice(n_actions, p=action_probs.numpy()[0])

# Compute cumulative rewards
def compute_cumulative_rewards(rewards, gamma=0.99):
    cumulative_rewards = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        cumulative_rewards[t] = running_add
    return cumulative_rewards

# Update policy network
def update_policy(states, actions, rewards):
    cumulative_rewards = compute_cumulative_rewards(rewards)
    with tf.GradientTape() as tape:
        state_inputs = tf.one_hot(states, n_states)
        action_probs = model(state_inputs)
        action_masks = tf.one_hot(actions, n_actions)
        log_probs = tf.reduce_sum(action_masks * tf.math.log(action_probs + 1e-8), axis=1)
        loss = -tf.reduce_mean(log_probs * cumulative_rewards)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# -------------------------------
# Policy Gradient training loop
# -------------------------------
for episode in range(1000):
    states = []
    actions = []
    rewards_episode = []

    state = np.random.randint(0, n_states)
    done = False
    while not done:
        action = get_action(state)
        next_state = np.random.randint(0, n_states)
        reward = rewards[next_state]

        states.append(state)
        actions.append(action)
        rewards_episode.append(reward)

        state = next_state
        if next_state in {24, 12}:
            done = True

    update_policy(states, actions, rewards_episode)

print("Policy Gradient training completed.")
