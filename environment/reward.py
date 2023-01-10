from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils import math
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.common_values import BALL_RADIUS, CAR_MAX_SPEED
import numpy as np

from rlgym.utils.reward_functions.common_rewards import LiuDistancePlayerToBallReward

liu_distance = LiuDistancePlayerToBallReward()


class CustomRewardFunction(RewardFunction):
  def reset(self, initial_state: GameState):
      pass

  def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
      # Compensate for inside of ball being unreachable (keep max reward at 1)
      dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS

      # print(np.exp(-0.5 * dist / CAR_MAX_SPEED))

      return np.exp(-0.5 * dist / CAR_MAX_SPEED)