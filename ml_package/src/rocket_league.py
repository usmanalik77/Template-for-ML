import rlgym
from stable_baselines3 import PPO
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils import math
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.terminal_conditions import TerminalCondition

import numpy as np

while True:
    obs = env.reset()
    done = False

    while not done:
      #Here we sample a random action. If you have an agent, you would get an action from it here.
      action = env.action_space.sample() 
      
      next_obs, reward, done, gameinfo = env.step(action)
      
      obs = next_obs


#Make the default rlgym environment
env = rlgym.make()

#Initialize PPO from stable_baselines3
model = PPO("MlpPolicy", env=env, verbose=1)

#Train our agent!
model.learn(total_timesteps=int(1e6))

###Reward Functions
class SpeedReward(RewardFunction):
  def reset(self, initial_state: GameState):
    pass

  def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
    linear_velocity = player.car_data.linear_velocity
    reward = math.vecmag(linear_velocity)
    
    return reward
    
  def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
    return 0
  

env = rlgym.make(reward_fn=SpeedReward())
#Training loop goes here


### Observation Builders

class CustomObsBuilder(ObsBuilder):
  def reset(self, initial_state: GameState):
    pass

  def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
    obs = []
    obs += state.ball.serialize()
    
    for player in state.players:
      obs += player.car_data.serialize()
    
    return np.asarray(obs, dtype=np.float32)
  

### Terminal Conditions

env = rlgym.make(obs_builder=CustomObsBuilder())
#Training loop goes here

class CustomTerminalCondition(TerminalCondition):
  def reset(self, initial_state: GameState):
    pass

  def is_terminal(self, current_state: GameState) -> bool:
    return current_state.last_touch != -1
  
env = rlgym.make(terminal_conditions=[CustomTerminalCondition()])
#Training loop goes here