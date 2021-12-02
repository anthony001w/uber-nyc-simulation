import pandas as pd
from sodapy import Socrata
import sys, pygame, os
import numpy as np
from joblib import dump,load
from collections import deque
from shapely.geometry import Polygon, Point
from tqdm import tqdm
from city_elements import *
from city import *
from event_list import *

SCREEN_SIZE = (1200,800)
FPS = 30
SPEED_OF_SIM = 30
DRIVER_MOVEMENT_FILENAME = '../output_d12k_50repl/driver_history_example'

class DriverAnimation:

	def __init__(self, animation_list):
		self.animations = animation_list
		self.center = np.array([0,0])
		self.speed = None
		self.current_animation_index = 0

	#this function handles updating the position of the driver's rectangle
	def update(self):

		if self.current_animation_index >= len(self.animations):
			pass
		else:
			curr_anim = self.animations[self.current_animation_index]
			if curr_anim.frames_covered == 0:

				#set the center and direction
				self.center = curr_anim.start
				self.speed = curr_anim.velocity
				curr_anim.update()

			elif curr_anim.frames_covered > curr_anim.total_frames:

				#reset the current animation index
				#reset the previous animation's frames covered
				curr_anim.reset()
				self.center = curr_anim.end 
				self.current_animation_index += 1
				self.speed = None

			else:

				#change the position of the center of the driver rectangle
				self.center += self.speed
				curr_anim.update()

		return self.center

	def has_passenger(self):
		if self.current_animation_index < len(self.animations):
			return self.animations[self.current_animation_index].passenger
		else:
			return False

	def get_time(self):
		if self.current_animation_index < len(self.animations):
			return self.animations[self.current_animation_index].current_time
		else:
			return None

class Animation:

	def __init__(self, start_time_system, end_time_system, start_pos, end_pos, velocity, approx_frames, passenger):
		self.start_time = start_time_system
		self.end_time = end_time_system
		self.current_time = start_time_system #in minutes
		self.start = start_pos #tuple/np.array
		self.end = end_pos #tuple/np.array
		self.velocity = velocity #tuple/np.array
		self.total_frames = approx_frames #float number of frames
		self.frames_covered = 0
		self.passenger = passenger #T/F

	def reset(self):
		self.frames_covered = 0
		self.current_time = self.start_time

	def update(self):
		self.frames_covered += 1
		if self.total_frames == 0.0001:
			self.current_time = self.start_time
		else:
			self.current_time += (self.end_time - self.start_time)/self.total_frames

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
    
    return np.c_[np.floor(bbox_dim[0] * np.abs(merc[:,0] - left) / total_dim[0]), 
                 np.floor(bbox_dim[1] * np.abs(top - merc[:,1]) / total_dim[1])], polygon_coords[1]

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
	for coord_list, zone_id in xy_pixel_polygons:
		if zone_id not in zone_polygon_dict:
			zone_polygon_dict[zone_id] = [Polygon(coord_list)]
		else:
			zone_polygon_dict[zone_id].append(Polygon(coord_list))
	return zone_polygon_dict

def extract_animations(dm, starting_position, polygon_dict, scaling = SPEED_OF_SIM, fps = FPS):
    current_position = random_point_within(starting_position[1], polygon_dict)
    current_time = starting_position[0]
    animation_list = []

    #dm alternates between (start of movement to location) (end of movement to location)
    #cut the list into (departures) (arrivals) and iterate
    for i in range(0,len(dm) // 2,2):
    	d = dm[i]
    	a = dm[i + 1]
    	start_pos = current_position
    	end_pos = random_point_within(a[1], polygon_dict)
    	v, f = calc_velocity(current_position, current_position, current_time, d[0])
    	animation_list.append(Animation(current_time, d[0], current_position, current_position, v, f, False))
    	v, f = calc_velocity(start_pos, end_pos, d[0], a[0])
    	animation_list.append(Animation(d[0], a[0], start_pos, end_pos, v, f, d[-1] is not None))
    	current_position = end_pos
    	current_time = a[0]

    return animation_list

#return a random point within the zone given a polygon dictionary
def random_point_within(zone_id, polygon_dict):

	#first choose a random polygon from the ones with the matching zone id
	poly = polygon_dict[zone_id][np.random.randint(len(polygon_dict[zone_id]))]

	#then select a random point using acceptance/rejection sampling
	min_x, min_y, max_x, max_y = poly.bounds

	num_tries = 30

	xs = np.random.uniform(min_x, max_x, num_tries)
	ys = np.random.uniform(min_y, max_y, num_tries)

	for i in range(num_tries):
		p = [xs[i], ys[i]]
		random_point = Point(p)
		if (random_point.within(poly)):
			return np.floor(np.array(p))

	return np.floor(np.array([min_x, min_y]))
		

#returns the velocity vector and how many frames it should go for
def calc_velocity(start_pos, end_pos, stime, etime, scaling = SPEED_OF_SIM, fps = FPS):

	frames = (etime - stime) / scaling * fps

	if frames == 0:
		frames = 0.0001

	velocity = 1/frames * (end_pos - start_pos)

	return velocity, frames

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
drivers = load(DRIVER_MOVEMENT_FILENAME)

#start with just the first driver movement

"""to convert movement data into an animation, maybe make a dot that moves across the screen?
   if it's at 60 fps, then gotta determine the vector + speed of movement from point a to b
   also need some rate to speed up how fast things are going

   direction = scaling / fps * (time end - time start) * [x end - x start, y end - y start]
"""

import time

def driver_to_animation(driver, z = zone_dict):
	m = driver.movement_history
	start_pos = m[0]
	movements = m[1:]
	animations = extract_animations(movements, start_pos, zone_dict)
	return DriverAnimation(animations)

driver_animations = [driver_to_animation(d) for d in tqdm(drivers, position = 0, leave = True)]

#initalize screen and stuff
pygame.init()
size = width, height = SCREEN_SIZE

driver_with_passenger = pygame.image.load('passenger.png')
driver_with_passenger = pygame.transform.scale(driver_with_passenger, (2,2))

driver_without_passenger = pygame.image.load('nopassenger.png')
driver_without_passenger = pygame.transform.scale(driver_without_passenger, (3,3))
driver_rectangles = [driver_without_passenger.get_rect() for i in range(len(driver_animations))]

screen = pygame.display.set_mode(size)

clock = pygame.time.Clock()
sys_time = 0
font = pygame.font.SysFont('Trebuchet MS', 20)
while True:

	clock.tick(FPS)

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			quit()

	#showing animations
	screen.fill([160, 160, 160])
	draw_bg(xy_pixel_polygons, screen)

	#animation handling
	for driver, rect in zip(driver_animations, driver_rectangles):
		rect.center = driver.update()
		animation_time = driver.get_time()
		if animation_time is not None and animation_time >= sys_time:
			sys_time = animation_time
		if driver.has_passenger():
			screen.blit(driver_with_passenger, rect)
		else:
			screen.blit(driver_without_passenger, rect)

	hour_font = font.render('Hour:{0:02}'.format(round(sys_time // 60)), 1, (0,0,0))
	minute_font = font.render('Minute:{0:02}'.format(round(sys_time % 60)), 1, (0,0,0))
	hour_rect = hour_font.get_rect()
	hour_rect.center = (40,40)
	minute_rect = minute_font.get_rect()
	minute_rect.center = (160,40)
	screen.blit(hour_font, hour_rect)
	screen.blit(minute_font, minute_rect)

	pygame.display.flip()