from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, CEILING_Z, SIDE_WALL_X, BACK_WALL_Y, BLUE_GOAL_CENTER
import numpy as np

SPEED_SCALE = 0.8

class CustomStateSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
    
        # Set up our desired spawn location and orientation. Here, we will only change the yaw, leaving the remaining orientation values unchanged.
        desired_car_pos = [BLUE_GOAL_CENTER[0],BLUE_GOAL_CENTER[1],17] #x, y, z
        desired_yaw = np.pi/2
        
        # Loop over every car in the game.
        for car in state_wrapper.cars:
            if car.team_num == BLUE_TEAM:
                pos = desired_car_pos
                yaw = desired_yaw
                
            elif car.team_num == ORANGE_TEAM:
                # We will invert values for the orange team so our state setter treats both teams in the same way.
                pos = [-1*coord for coord in desired_car_pos]
                yaw = -1*desired_yaw
                
            # Now we just use the provided setters in the CarWrapper we are manipulating to set its state. Note that here we are unpacking the pos array to set the position of 
            # the car. This is merely for convenience, and we will set the x,y,z coordinates directly when we set the state of the ball in a moment.
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = 0.33
            
        # Now we will spawn the ball in the center of the field, floating in the air.
        random_ball_loc = (np.random.randint(-700, 700), np.random.randint(-BACK_WALL_Y+1500, -BACK_WALL_Y+2000), 100)

        state_wrapper.ball.set_pos(x=random_ball_loc[0], y=random_ball_loc[1], z=random_ball_loc[2])
        
        state_wrapper.ball.set_lin_vel(x=(BLUE_GOAL_CENTER[0] + np.random.randint(-700, 700) - random_ball_loc[0])*SPEED_SCALE, y=(BLUE_GOAL_CENTER[1] - random_ball_loc[1])*SPEED_SCALE, z=(BLUE_GOAL_CENTER[2] - random_ball_loc[2])*SPEED_SCALE )

class RandomStateSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
        
        random_ball_target = np.array([(np.random.rand()*2-1)*SIDE_WALL_X, (np.random.rand()*2-1)*BACK_WALL_Y, (np.random.rand())*CEILING_Z])

        # Loop over every car in the game.
        for car in state_wrapper.cars:
            pos = np.array([(np.random.rand()*2-1)*SIDE_WALL_X, (np.random.rand()*2-1)*BACK_WALL_Y, (np.random.rand())*CEILING_Z])
            rot = (np.random.rand(3)*2-1)*np.pi

            car.set_pos(*pos)
            car.set_rot(*rot)
            car.boost = np.random.rand()*0.3 + 0.7

            if np.random.rand() > 0.65:
                random_ball_target = pos
            
        # Now we will spawn the ball in the center of the field, floating in the air.
        random_ball_loc = np.maximum(np.minimum((random_ball_target + (np.random.rand(3)*2-1)*1500), np.array([SIDE_WALL_X, BACK_WALL_Y, CEILING_Z])), np.array([-SIDE_WALL_X, -BACK_WALL_Y, 0]))
        random_ball_vel = (random_ball_target - random_ball_loc)*np.random.rand()*7

        state_wrapper.ball.set_pos(*random_ball_loc)
        
        state_wrapper.ball.set_lin_vel(*random_ball_vel)