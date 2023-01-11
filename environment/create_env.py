import rlgym
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from environment.statesetter import RandomStateSetter
from environment.reward import CustomRewardFunction
from environment.actions import CustomAction
from environment.observer import ModifiedDefaultObs, MinimalCarObs

tick_skip = 4
physics_ticks_per_second = 120
ep_len_seconds = 8

max_steps = int(round(ep_len_seconds * physics_ticks_per_second / tick_skip))

def make(spawn_opponents=True, game_speed=100):
    return rlgym.make(spawn_opponents=spawn_opponents, tick_skip=tick_skip, team_size=1, game_speed = game_speed, 
        obs_builder=MinimalCarObs(), reward_fn=CustomRewardFunction(), state_setter=RandomStateSetter(), 
        action_parser=CustomAction(), terminal_conditions=[GoalScoredCondition(), TimeoutCondition(max_steps)])