import rlgym
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from environment.statesetter import RandomStateSetter
from environment.reward import CustomRewardFunction
from environment.actions import CustomAction
from environment.observer import ModifiedDefaultObs, MinimalCarObs

default_tick_skip = 1
physics_ticks_per_second = 120
ep_len_seconds = 10

max_steps = int(round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

def make():
    return rlgym.make(spawn_opponents=True, tick_skip=1, team_size=1, game_speed = 100, 
        obs_builder=MinimalCarObs(), reward_fn=CustomRewardFunction(), state_setter=RandomStateSetter(), 
        action_parser=CustomAction(), terminal_conditions=[GoalScoredCondition(), TimeoutCondition(max_steps)])