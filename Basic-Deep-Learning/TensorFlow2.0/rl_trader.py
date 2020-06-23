import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Let's use AAPL (Apple), MSI (Motorola), SBUX (Starbucks)
def get_data():
  """
  returns a T x 3 list of stock prices
  each row is a different stock
    0 = AAPL
    1 = MSI
    2 = SBUX

  Returns:
      dataframe

  """  
  
  df = pd.read_csv('aapl_msi_sbux.csv')
  return df.values


def get_scaler(env):
  """
  return scikit-learn scaler object to scale the states
  Note: you could also populate the replay buffer here
        one thing you could do to make this more accurate is 
        to run this for multiple episodes

  Args:
      env (object): [description]

  Returns:
      scikit-learn scaler object

  """ 
  states = []
  for _ in range(env.n_step):
    action = np.random.choice(env.action_space)
    state, reward, done, info = env.step(action)
    states.append(state)
    if done:
      break

  scaler = StandardScaler()
  scaler.fit(states)
  return scaler


def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)


def mlp(input_dim, n_action, n_hidden_layers=1, hidden_dim=32):
  """
  A multi-layer perceptron, you can make a more complicated NN by yourself

  Args:
      input_dim (int): 
      n_action (int): number of actions agent could take
      n_hidden_layers (int, optional): number of hidden layers. Defaults to 1.
      hidden_dim (int, optional): numbe of neurons for each hidden layer. Defaults to 32.

  Returns:
      model (TF NN model)

  """  
  # input layer
  i = Input(shape=(input_dim,))
  x = i

  # hidden layers
  for _ in range(n_hidden_layers):
    x = Dense(hidden_dim, activation='relu')(x)
  
  # final layer, regression 
  x = Dense(n_action)(x)

  # make the model
  model = Model(i, x)
  model.compile(loss='mse', optimizer='adam')
  print((model.summary()))
  return model



### The experience replay memory ###
class ReplayBuffer:
  def __init__(self, obs_dim, act_dim, size):
    """
    Args:
        obs_dim (int): state dimension
        act_dim (int): 
        size (int): buffer size

    """    
    self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32) # store state
    self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32) # store next state
    self.acts_buf = np.zeros(size, dtype=np.uint8)
    self.rews_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.uint8) # True of False, i.e. 1 or 0
    self.ptr, self.size, self.max_size = 0, 0, size


  def store(self, obs, act, rew, next_obs, done):
    """
    store everything we need to buffer

    Args:
        obs (np.array): current state vector
        act (int): index of the action
        rew (int): rewards
        next_obs (np.array): next state vector
        done (int or bool): 0 or 1

    """
    self.obs1_buf[self.ptr] = obs
    self.obs2_buf[self.ptr] = next_obs
    self.acts_buf[self.ptr] = act
    self.rews_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done
    self.ptr = (self.ptr+1) % self.max_size
    self.size = min(self.size+1, self.max_size)


  def sample_batch(self, batch_size=32):
    """
    sample a batch of training data

    Args:
        batch_size (int, optional): Defaults to 32.

    Returns:
        a dictionary that contains (current_state, next_state, action, reward, done)

    """    
    idxs = np.random.randint(0, self.size, size=batch_size)
    return dict(s=self.obs1_buf[idxs],
                s2=self.obs2_buf[idxs],
                a=self.acts_buf[idxs],
                r=self.rews_buf[idxs],
                d=self.done_buf[idxs])



class MultiStockEnv:
  """
  A 3-stock trading environment.

  State: vector of size 7 (n_stock * 2 + 1)
    - # shares of stock 1 owned
    - # shares of stock 2 owned
    - # shares of stock 3 owned
    - price of stock 1 (using daily close price)
    - price of stock 2
    - price of stock 3
    - cash owned (can be used to purchase more stocks)

  Action: categorical variable with 27 (3^3) possibilities
    - for each stock, you can:
    - 0 = sell
    - 1 = hold
    - 2 = buy

  """
  def __init__(self, data, initial_investment=20000):
    """
    Args:
        data (dataframe): 
        initial_investment (int, optional): intial cash we have. Defaults to 20000.

    """    
    # data
    self.stock_price_history = data
    self.n_step, self.n_stock = self.stock_price_history.shape

    # instance attributes
    self.initial_investment = initial_investment
    self.cur_step = None
    self.stock_owned = None
    self.stock_price = None
    self.cash_in_hand = None
    self.action_space = np.arange(3 ** self.n_stock)

    # action permutations
    # returns a nested list with elements like:
    # [0,0,0]
    # [0,0,1]
    # [0,0,2]
    # [0,1,0]
    # [0,1,1]
    # etc.
    # 0 = sell
    # 1 = hold
    # 2 = buy
    # we should have 3^3 = 27 different actions
    self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

    # calculate size of state
    self.state_dim = self.n_stock * 2 + 1
    self.reset()

  ############## public function ##############
  def reset(self):
    self.cur_step = 0
    self.stock_owned = np.zeros(self.n_stock)
    self.stock_price = self.stock_price_history[self.cur_step]
    self.cash_in_hand = self.initial_investment
    return self._get_obs()


  def step(self, action):
    """
    take the action, make all the movements needed 

    Args:
        action (int): index of the actions

    Returns:
        tuple: (state vector, reward, done, info)

    """    
    assert action in self.action_space

    # get current value before performing the action
    prev_val = self._get_val()
    # update price, i.e. go to the next day
    self.cur_step += 1
    self.stock_price = self.stock_price_history[self.cur_step]
    # perform the trade
    self._trade(action)

    # get the new value after taking the action
    cur_val = self._get_val()
    # reward is the increase in porfolio value
    reward = cur_val - prev_val
    # done if we have run out of data
    done = self.cur_step == self.n_step - 1

    # store the current value of the portfolio here
    info = {'cur_val': cur_val}

    # conform to the Gym API
    return self._get_obs(), reward, done, info


  ############## private function ##############
  def _get_obs(self):  
    """
    get state vector

    Returns:
        np.array

    """    
    obs = np.empty(self.state_dim)
    obs[ : self.n_stock] = self.stock_owned
    obs[self.n_stock : 2*self.n_stock] = self.stock_price
    obs[-1] = self.cash_in_hand
    return obs


  def _get_val(self):
    # return current value of the portfolio
    return self.stock_owned.dot(self.stock_price) + self.cash_in_hand


  def _trade(self, action):
    """
    Trading logic

    we should index the action we want to perform:
      0 = sell
      1 = hold
      2 = buy
    e.g. [2,1,0] means:
      buy first stock
      hold second stock
      sell third stock

    Args:
        action (int): index of the action we want to perform 

    """    
    action_vec = self.action_list[action]

    # determine which stocks to buy or sell
    sell_index = [] # stores index of stocks we want to sell
    buy_index = [] # stores index of stocks we want to buy
    for i, a in enumerate(action_vec):
      if a == 0:
        sell_index.append(i)
      elif a == 2:
        buy_index.append(i)

    # (1) sell any stocks we want to sell
    # (2) then buy any stocks we want to buy
    if sell_index:
      # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
      for i in sell_index:
        self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
        self.stock_owned[i] = 0
    
    if buy_index:
      # NOTE: when buying, we will loop through each stock we want to buy,
      #       and buy one share at a time until we run out of cash
      can_buy = True
      while can_buy:
        for i in buy_index:
          if self.cash_in_hand > self.stock_price[i]: # if we have enough cash to buy one share of this stock
            self.stock_owned[i] += 1 # buy one share
            self.cash_in_hand -= self.stock_price[i]
          else:
            can_buy = False



class DQNAgent(object):
  """

  Args:
      object ([type]): [description]

  """  
  def __init__(self, state_size, action_size):
    """

    Args:
        state_size (int): length of the state vector
        action_size (int): number of actions in total

    """    
    self.state_size = state_size
    self.action_size = action_size
    self.memory = ReplayBuffer(state_size, action_size, size=500)
    self.gamma = 0.95  # discount rate

    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.model = mlp(state_size, action_size) # create NN 


  def update_replay_memory(self, state, action, reward, next_state, done):
    self.memory.store(state, action, reward, next_state, done)


  def act(self, state):
    """
    either return random action or greedy action

    Args:
        state (np.array) : current state vector

    Returns:
        int, index of the action to take

    """    
    if np.random.rand() <= self.epsilon: # perform random action
      return np.random.choice(self.action_size)

    act_values = self.model.predict(state)
    return np.argmax(act_values[0])  # returns action


  def replay(self, batch_size=32):
    """
    This is the function that does the learning

    Args:
        batch_size (int, optional): Defaults to 32.

    """    
    # first check if replay buffer contains enough data
    if self.memory.size < batch_size:
      return

    # sample a batch of data from the replay memory
    mini_batch = self.memory.sample_batch(batch_size) # return a dictionary
    states = mini_batch['s'] # matrix: n x D, n=batch_size, D=state_size
    actions = mini_batch['a'] # matrix: n x K, n=batch_size, K=action_size
    rewards = mini_batch['r'] # vector: n, n=batch_size
    next_states = mini_batch['s2'] # matrix: n x D, n=batch_size, D=state_size
    done = mini_batch['d'] # vector: n, n=batch_size

    # Calculate the tentative target: Q(s',a)
    # done = 0 or 1
    # 0: not the terminal state
    target = rewards + (1 - done) * self.gamma * np.amax(self.model.predict(next_states), axis=1)

    # With the Keras API, the target (usually) must have the same shape as the predictions!
    # However, we only need to update the network for the actions which were actually taken.
    # We can accomplish this by setting the target to be equal to the prediction for all values!
    # Then, only change the targets for the actions taken.
    # Q(s,a)
    target_full = self.model.predict(states) # matrix: n x D, n=batch_size, D=state_size
    target_full[np.arange(batch_size), actions] = target

    # Run one training step
    # print(f'states shape is: {states.shape}')
    # print(f'target_full shape is: {target_full.shape}')
    self.model.train_on_batch(x=states, y=target_full)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay


  def load(self, name):
    self.model.load_weights(name)


  def save(self, name):
    self.model.save_weights(name)


def play_one_episode(agent, env, is_train):
  """

  Args:
      agent ([type]): [description]
      env ([type]): [description]
      is_train (bool): [description]

  Returns:
      [type]: [description]

  """  
  # note: after transforming states are already 1xD
  state = env.reset()
  state = scaler.transform([state])
  done = False

  while not done:
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    next_state = scaler.transform([next_state])
    if is_train == 'train':
      agent.update_replay_memory(state, action, reward, next_state, done)
      agent.replay(batch_size)

    state = next_state

  return info['cur_val']



if __name__ == '__main__':

  # config
  models_folder = 'rl_trader_models'
  rewards_folder = 'rl_trader_rewards'
  num_episodes = 2000
  batch_size = 32
  initial_investment = 20000

  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--mode', type=str, required=True,
                      help='either "train" or "test"')
  args = parser.parse_args()

  maybe_make_dir(models_folder)
  maybe_make_dir(rewards_folder)

  data = get_data()
  n_timesteps, n_stocks = data.shape

  n_train = n_timesteps // 2
  train_data = data[:n_train]
  test_data = data[n_train:]

  ## create environment and agent
  env = MultiStockEnv(train_data, initial_investment)
  state_size = env.state_dim
  action_size = len(env.action_space)
  agent = DQNAgent(state_size, action_size)
  scaler = get_scaler(env)

  # store the final value of the portfolio (end of episode)
  portfolio_value = []

  if args.mode == 'test':
    # then load the previous scaler
    with open(f'{models_folder}/scaler.pkl', 'rb') as f:
      scaler = pickle.load(f)

    # remake the env with test data
    env = MultiStockEnv(test_data, initial_investment)

    # make sure epsilon is not 1!
    # no need to run multiple episodes if epsilon = 0, it's deterministic
    agent.epsilon = 0.01

    # load trained weights
    agent.load(f'{models_folder}/dqn.h5')

  # play the game num_episodes times
  for e in range(num_episodes):
    t0 = datetime.now()
    val = play_one_episode(agent, env, args.mode)
    dt = datetime.now() - t0
    print(f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}")
    portfolio_value.append(val) # append episode end portfolio value

  # save the weights when we are done
  if args.mode == 'train':
    # save the DQN
    agent.save(f'{models_folder}/dqn.h5')

    # save the scaler
    with open(f'{models_folder}/scaler.pkl', 'wb') as f:
      pickle.dump(scaler, f)


  # save portfolio value for each episode
  np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)

  ######### plot the reward #########
  portfolio_value = np.array(portfolio_value)
  print(f"average reward: {portfolio_value.mean():.2f}, min: {portfolio_value.min():.2f}, max: {portfolio_value.max():.2f}")

  plt.hist(portfolio_value, bins=20)
  plt.title(args.mode)
  plt.show()

  # python rl_trader.py -m train
  # python rl_trader.py -m test