import numpy as np
import pandas as pd

def vf(df, st, et, sx, sy, ex, ey, scaling, fps):
	frames = (df[et] - df[st]) / scaling * fps
	frames[frames == 0] = 0.0001
	frames = frames.values
	direction = (df[[ex, ey]].values - df[[sx, sy]].values).astype('float')
	velocity = (1/frames * direction.T).T
	return velocity, frames

class DriverAnimation:

    def __init__(self, position_df, SCALING, FPS):

        self.movement_df = position_df
        self.scaling = SCALING
        self.fps = FPS

        vs, fs = vf(self.movement_df, 'start_time', 'end_time', 'startx','starty','endx','endy',self.scaling, self.fps)
        self.movement_df[['vx','vy']] = vs
        self.movement_df['frames'] = fs

        #immediately calculate initial positions, set the indices
        self.animation_indices = self.movement_df.groupby('driver_id').apply(lambda g: g.index[0]).values
        self.last_indices = self.movement_df.groupby('driver_id').apply(lambda g: g.index[-1]).values
        self.positions = self.movement_df.loc[self.animation_indices][['startx', 'starty']].values
        self.updated_frames = np.zeros(len(self.animation_indices))
        self.driver_count = len(self.updated_frames)
        self.finished_animation = np.zeros(self.driver_count, dtype = bool)

    def update(self):

        #updating takes a few steps
        #need to check whether the frame count surpasses the total frames, if it does then reset
        curr_animations = self.movement_df.loc[self.animation_indices]
        stay_on_current = self.updated_frames <= curr_animations['frames'].values

        self.updated_frames = self.updated_frames * stay_on_current
        self.animation_indices = self.animation_indices + ~stay_on_current
        self.finished_animation = self.finished_animation | (self.animation_indices > self.last_indices)
        fanimation = self.finished_animation.reshape((self.driver_count, 1))
        self.animation_indices = np.minimum(self.animation_indices, self.last_indices)

        stay_on_current = stay_on_current.reshape((self.driver_count, 1))
        
        #grab next animations
        next_animations = self.movement_df.loc[self.animation_indices]

        #set initial positions for the new animations
        new_pos = next_animations[['startx','starty']].values * ~stay_on_current

        #last positions
        last_pos = next_animations[['endx','endy']].values * fanimation

        #add the position vectors for the old animations
        velocity = next_animations[['vx','vy']].values * stay_on_current
        
        self.positions = (self.positions * stay_on_current) + (new_pos + velocity) * ~fanimation + last_pos * fanimation

        self.updated_frames += 1

        #should return the positions, whether or not the driver has passenger/is moving, and the time

        return self.positions, next_animations['is_moving'], next_animations['has_passenger'], self.finished_animation, next_animations.start_time.max()