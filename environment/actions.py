import numpy as np
import gym.spaces
from rlgym.utils.gamestates import GameState
from rlgym.utils.action_parsers import ActionParser


class CustomAction(ActionParser):

    def __init__(self, n_bins=3):
        super().__init__()
        assert n_bins % 2 == 1, "n_bins must be an odd number"
        self._n_bins = n_bins

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(-1, 1, (1, 8))

    def parse_actions(self, actions: np.ndarray, state: GameState) -> np.ndarray:
        if type(actions) != np.ndarray:
            actions = np.asarray(actions)

        if len(actions.shape) == 1:
            actions = actions.reshape((-1, 8))
        elif len(actions.shape) > 2:
            raise ValueError('{} is not a valid action shape'.format(actions.shape))
        
        return actions

class DiscreteAction(ActionParser):
    """
    Simple discrete action space. All the analog actions have 3 bins by default: -1, 0 and 1.
    """

    def __init__(self, n_bins=3):
        super().__init__()
        assert n_bins % 2 == 1, "n_bins must be an odd number"
        self._n_bins = n_bins

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.MultiDiscrete([self._n_bins] * 5 + [2] * 3)

    def parse_actions(self, actions: np.ndarray, state: GameState) -> np.ndarray:
        actions = actions.reshape((-1, 8))

        # map all ternary actions from {0, 1, 2} to {-1, 0, 1}.
        actions[..., :5] = actions[..., :5] / (self._n_bins // 2) - 1

        return actions