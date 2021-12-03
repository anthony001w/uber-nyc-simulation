import pandas as pd
from sodapy import Socrata
import sys, pygame, os
import numpy as np
from joblib import dump,load, Parallel, delayed
from collections import deque
from shapely.geometry import Polygon, Point
from tqdm import tqdm
from city_elements import *
from city import *
from event_list import *
from matplotlib.path import Path

SCREEN_SIZE = (1200,800)
if len(sys.argv) == 4:
	FPS = int(sys.argv[1])
	SPEED_OF_SIM = int(sys.argv[2])
	FOLDER = sys.argv[3]

else:
	FPS = 60
	SPEED_OF_SIM = 15
	FOLDER = input('Enter test folder name:')

DRIVER_MOVEMENT_FILENAME = f'../{FOLDER}/driver_histories_parquet'

class DriverAnimation:
    
    sx, sy, ex, ey = 7, 8, 9, 10
    st, et, vx, vy = 0, 1, 11, 12
    is_move, has_pass, f = 4,5,13
    
    def __init__(self, movement_matrix):
        self.movements = movement_matrix
        self.center = np.array([movement_matrix[0][self.sx], movement_matrix[0][self.sy]])
        self.prev_center = self.center
        self.current_movement_index = 0
        self.frame_count = 0
        
    def update(self):
        self.prev_center = self.center
        if self.current_movement_index < self.movements.shape[0]:
            m = self.movements[self.current_movement_index]
            if self.frame_count == 0:
                self.center = np.array([m[self.sx], m[self.sy]])
                self.frame_count += 1
            elif self.frame_count > m[self.f]:
                self.center = np.array([m[self.ex], m[self.ey]])
                self.current_movement_index += 1
                self.frame_count = 0
            else:
                self.center += np.array([m[self.vx], m[self.vy]])
                self.frame_count += 1
        return self.center, self.prev_center
    
    #0 for not moving
    #1 for moving with passenger
    #2 for moving without passenger
    def state(self):
        if self.current_movement_index >= self.movements.shape[0]:
            return 0
        else:
            m = self.movements[self.current_movement_index]
            if m[self.has_pass]:
                return 1
            elif m[self.is_move]:
                return 2
            else:
                return 0
    
    def get_time(self):
        if self.current_movement_index >= self.movements.shape[0]:
            return None
        else:
            m = self.movements[self.current_movement_index]
            return m[self.st] + (m[self.st] - m[self.et])/m[self.f] * self.frame_count

def add_polygons(lists, poly_list, region_id):
    
    if len(lists) > 1 and np.array([len(l) == 2 for l in lists]).all():
        poly_list.append([lists, region_id])
    else:
        for l in lists:
            add_polygons(l, poly_list, region_id)

def merc_from_arrays(lats, lons):
    r_major = 6378137.000
    x = r_major * np.radians(lons)
    scale = x/lons
    y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + lats * (np.pi/180.0)/2.0)) * scale
    return np.c_[x, y]

#converts (lon,lat) from socrata -> pixel positions of the bounding box
def convert_coords(polygon_coords, left, top, total_dim, bbox_dim = SCREEN_SIZE):

    p = np.array(polygon_coords[0])
    merc = merc_from_arrays(p[:,1], p[:,0])

    bbox_coords = np.c_[np.floor(bbox_dim[0] * np.abs(merc[:,0] - left) / total_dim[0]), 
                  	np.floor(bbox_dim[1] * np.abs(top - merc[:,1]) / total_dim[1])]

    return bbox_coords, polygon_coords[1]

def extract_zones_as_polygons(taxi_zone_information):
	#extract all the coordinate points first (taking into account weird shapes)
	polygons = []
	for i,z in taxi_zone_information.iterrows():
	    p = z['the_geom']['coordinates']
	    
	    add_polygons(p, polygons, int(z.objectid))
	    
	#get the (lon, lat) coordinates of every single polygon in 1 list
	coords = np.concatenate([p[0] for p in polygons])

	#mercator projection applied to every coordinate
	mercs = merc_from_arrays(coords[:,1], coords[:,0])

	#get the extreme ends
	left, bottom, right, top = mercs[:,0].argmin(), mercs[:,1].argmin(), mercs[:,0].argmax(), mercs[:,1].argmax()
	
	#calculate the distances from end to end (top-bottom and left-right)
	d_x = np.abs(mercs[left] - mercs[right])[0]
	d_y = np.abs(mercs[top] - mercs[bottom])[1]

	#apply the conversion using the information above
	xy_pixel_polygons = []
	for p in polygons:
		to_append = convert_coords(p, mercs[left][0], mercs[top][1], (d_x, d_y))
		xy_pixel_polygons.append(to_append)

	return xy_pixel_polygons

def convert_polygons_for_sampling(xy_pixel_polygons):
	#store polygons as a dictionary of zone_id:polygon for faster sampling
	zone_polygon_dict = {}
	for coord_list, zone_id in tqdm(xy_pixel_polygons, position = 0, leave = True, desc = 'Zone Polygons Processed'):
		poly_info = (coord_list, Polygon(coord_list).area)
		if zone_id not in zone_polygon_dict:
			zone_polygon_dict[zone_id] = [poly_info]
		else:
			zone_polygon_dict[zone_id].append(poly_info)
	
	for zone in tqdm(zone_polygon_dict, position = 0, leave = True, desc = 'Best Polygons Chosen'):
		#only use the polygon with the most area
		best_polygon = max(zone_polygon_dict[zone], key = lambda p: p[1])
		zone_polygon_dict[zone] = Path(best_polygon[0])

	return zone_polygon_dict

def draw_bg(poly_list, s):
	for p in poly_list:
		pygame.draw.polygon(s, 0, p[0], width = 1)

""" script part,
	first get the taxi zone information from the api (fast)
	then create the zone outlines
	then add in the sequential driver movement information

"""

#polygon data only needs to be created once
if not os.path.exists('polygon_info'):
	with Socrata('data.cityofnewyork.us', 'vA3MfkSw5kKhpzNkitJkv5yFP') as client:
	    #gets the taxi zones (the geometry of each zone in the city + the zone name and borough)
	    taxi_zones =  pd.DataFrame.from_records(client.get('755u-8jsi'))

	xy_pixel_polygons = extract_zones_as_polygons(taxi_zones)
	zone_dict = convert_polygons_for_sampling(xy_pixel_polygons)

	dump([xy_pixel_polygons, zone_dict], 'polygon_info')
else:
	polygon_info = load('polygon_info')
	xy_pixel_polygons = polygon_info[0]
	zone_dict = polygon_info[1]

#load some provided driver movement data
driver_history = pd.read_parquet(DRIVER_MOVEMENT_FILENAME).reset_index(drop = True)

if not os.path.exists(f'../{FOLDER}/driver_generated_points'):

	extent_dict = {k:(v.get_extents().min, v.get_extents().max) for k,v in zone_dict.items()}

	def generate_points(zone_id, num_required, zone_dict = zone_dict, extent_dict = extent_dict):
	    bounds = extent_dict[zone_id]
	    path = zone_dict[zone_id]
	    generated_points = np.array([])
	    while len(generated_points) < num_required:
	        xs = np.random.uniform(bounds[0][0], bounds[1][0], size = num_required)
	        ys = np.random.uniform(bounds[0][1], bounds[1][1], size = num_required)
	        points = np.c_[xs,ys]
	        if len(generated_points) == 0:
	            generated_points = points
	        else:
	            generated_points = np.append(generated_points, points, axis = 0)
	    return generated_points[:num_required].round(2)

	#get the # of points needed to be generated for each zone
	#the zones that need (x,y) generation are the end zones after trips where the driver moves
	#and the beginning position of each driver
	need_positions_generated = driver_history[(driver_history.start_time == 0) | (driver_history.is_moving)]
	points_required = need_positions_generated.groupby('end_zone').start_time.count()

	g = []
	for i in tqdm(points_required.index, 
		position = 0, 
		leave = True, 
		desc = 'Points Generated per Zone'):
	    positions_generated = generate_points(i, points_required.loc[i])
	    pos_df = pd.DataFrame(positions_generated, columns = ['x','y'])
	    pos_df['end_zone'] = i
	    g.append(pos_df)
	g = pd.concat(g)

	position_dfs = []
	for group in tqdm(need_positions_generated.groupby(['end_zone']), 
		position = 0, 
		leave = True, 
		desc = 'Points Matched per Zone'):
	    ind = group[1].index
	    zone = group[1].end_zone.iloc[0]
	    
	    #create a new dataframe with the index and corresponding values from generated points
	    position_df = pd.DataFrame(g[g.end_zone == zone][['x','y']].values, index = ind, columns = ['x','y'])
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

	driver_history.to_parquet(f'../{FOLDER}/driver_generated_points')

driver_generated_points = pd.read_parquet(f'../{FOLDER}/driver_generated_points')

#generate velocity and frame information
def vf(df, st, et, sx, sy, ex, ey, scaling=SPEED_OF_SIM, fps=FPS):
	frames = (df[et] - df[st]) / scaling * fps
	frames[frames == 0] = 0.0001
	frames = frames.values
	direction = (df[[ex, ey]].values - df[[sx, sy]].values).astype('float')
	velocity = (1/frames * direction.T).T
	return velocity, frames

v, f = vf(driver_generated_points, 'start_time', 'end_time', 'startx','starty','endx','endy')
driver_generated_points[['vx','vy']] = v
driver_generated_points['frames'] = f

driver_animations = []
#for each driver, build a DriverAnimation object
for did in tqdm(driver_generated_points.driver_id.unique(),
	position = 0,
	leave = True,
	desc = 'Driver Animation Objects Created'):
	matrix = driver_generated_points[driver_generated_points.driver_id == did].values
	new_driver_animation = DriverAnimation(matrix)
	driver_animations.append(new_driver_animation)


#initalize screen and stuff
pygame.init()
size = width, height = SCREEN_SIZE
screen = pygame.display.set_mode(size)

driver_with_passenger = pygame.image.load('passenger.png')
driver_with_passenger = pygame.transform.scale(driver_with_passenger, (3,3))

driver_without_passenger = pygame.image.load('nopassenger.png')
driver_without_passenger = pygame.transform.scale(driver_without_passenger, (3,3))

driver_idle = pygame.image.load('idle.png')
driver_idle = pygame.transform.scale(driver_idle, (3,3))

driver_rectangles = [driver_without_passenger.get_rect() for i in range(len(driver_animations))]

layer1 = pygame.Surface(SCREEN_SIZE)
layer1.fill([225,225,225])
draw_bg(xy_pixel_polygons, layer1)

clock = pygame.time.Clock()
sys_time = 0
font = pygame.font.SysFont('Trebuchet MS', 20)
while True:

	clock.tick(FPS)

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			quit()

	screen.blit(layer1, (0,0))

	#animation handling
	for driver, rect in zip(driver_animations, driver_rectangles):
		rect.center, trail_center = driver.update()
		animation_time = driver.get_time()
		if animation_time is not None and animation_time >= sys_time:
			sys_time = animation_time
		if driver.state() == 1:
			screen.blit(driver_with_passenger, rect)
		elif driver.state() == 2:
			screen.blit(driver_without_passenger, rect)
		else:
			screen.blit(driver_idle, rect)

	hour_font = font.render('Hour:{0:02}'.format(round(sys_time // 60)), 1, (0,0,0))
	minute_font = font.render('Minute:{0:02}'.format(round(sys_time % 60)), 1, (0,0,0))
	hour_rect = hour_font.get_rect()
	hour_rect.center = (40,40)
	minute_rect = minute_font.get_rect()
	minute_rect.center = (160,40)
	screen.blit(hour_font, hour_rect)
	screen.blit(minute_font, minute_rect)

	pygame.display.flip()