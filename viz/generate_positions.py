import pandas as pd
import numpy as np
from matplotlib.path import Path 
import os
from tqdm import tqdm

def generate_points_random(zone_id, num_required, zone_dict, extent_dict):
	bounds = extent_dict[zone_id]
	path = zone_dict[zone_id]
	generated_points = np.array([])
	while len(generated_points) < num_required:
		xs = np.random.uniform(bounds[0][0], bounds[1][0], size = num_required*2)
		ys = np.random.uniform(bounds[0][1], bounds[1][1], size = num_required*2)
		points = np.c_[xs,ys]
		points_filtered = points[path.contains_points(points)]
		if len(generated_points) == 0:
			generated_points = points_filtered
		else:
			generated_points = np.append(generated_points, points_filtered, axis = 0)
	return generated_points[:num_required].round()

def generate_points_lines(zone_id, num_required, zone_dict, extent_dict):
    path = zone_dict[zone_id]
    v = path.vertices[0]
    return v * np.ones((num_required, 2))

def generate_positions(driver_movement_filenames, zone_dict, folder, mode = 'random'):

    filepath = f'../{folder}/driver_generated_points_{mode}'

    #load some provided driver movement data
    driver_history = pd.read_parquet(driver_movement_filenames).reset_index(drop = True)
    extent_dict = {k:(v.get_extents().min, v.get_extents().max) for k,v in zone_dict.items()}
    if mode == 'random':
        point_generation_function = generate_points_random
    elif mode == 'lines':
        point_generation_function = generate_points_lines

    if not os.path.exists(filepath):

        """get the # of points needed to be generated for each zone
        the zones that need (x,y) generation are the end zones after trips where the driver moves
        and the beginning position of each driver
        """
        need_positions_generated = driver_history[(driver_history.start_time == 0) | (driver_history.is_moving)]
        points_required = need_positions_generated.groupby('end_zone').start_time.count()

        #generates random positions (unassigned)
        positions = []
        for i in tqdm(points_required.index, 
            position = 0, 
            leave = True, 
            desc = 'Points Generated per Zone'):
            positions_generated = point_generation_function(i, points_required.loc[i], zone_dict = zone_dict, extent_dict=extent_dict)
            pos_df = pd.DataFrame(positions_generated, columns = ['x','y'])
            pos_df['end_zone'] = i
            positions.append(pos_df)
        positions = pd.concat(positions)

        #assigns the randomly generated positions to the necessary zones
        position_dfs = []
        for group in tqdm(need_positions_generated.groupby(['end_zone']), 
            position = 0, 
            leave = True, 
            desc = 'Points Matched per Zone'):
            ind = group[1].index
            zone = group[1].end_zone.iloc[0]
            
            #create a new dataframe with the index and corresponding values from generated points
            position_df = pd.DataFrame(positions[positions.end_zone == zone][['x','y']].values, index = ind, columns = ['x','y'])
            position_dfs.append(position_df)
        position_dfs = pd.concat(position_dfs).sort_index()

        #set the original dataframe's columns
        driver_history[['end_x','end_y']] = position_dfs

        #assuming the end zones are alternating NaN and non-null values
        #for every zone but the first set the null values to the non-null value above it
        start_end_dfs = []
        #the start x,y equals the end x,y but shifted up 1 (except for the first, which is the same as the end)
        for did, group in tqdm(driver_history.groupby('driver_id'), 
            position = 0, 
            leave = True, 
            desc = 'Driver Positions Processed'):
            first_move_end = group.iloc[0][['end_x','end_y']].values
            next_moves_end = group.iloc[1:][['end_x','end_y']].values
            next_moves_end[1::2] = next_moves_end[:-1:2]
            end_moves = np.r_[[first_move_end], next_moves_end]
            start_moves = np.r_[[first_move_end],end_moves[:-1]]
            sedf = pd.DataFrame(np.c_[start_moves, end_moves], 
                                index = group.index, 
                                columns = ['startx','starty','endx','endy'])
            start_end_dfs.append(sedf)
        start_end_dfs = pd.concat(start_end_dfs)

        driver_history[['startx','starty','endx','endy']] = start_end_dfs
        driver_history.drop(columns = ['end_x','end_y'],inplace = True)
        driver_history.to_parquet(filepath)

        return driver_history
    else:
        return pd.read_parquet(filepath)