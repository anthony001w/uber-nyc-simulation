import pandas as pd
import sys, pygame
from tqdm import tqdm
from city_elements import *
from city import *
from event_list import *
from DriverAnimation import DriverAnimation
from matplotlib.path import Path
from generate_polygon_info import *
from generate_positions import *

#changing the screen size argument messes with everything don't do it
SCREEN_SIZE = (1200,800)
if len(sys.argv) == 4:
	FPS = int(sys.argv[1])
	SPEED_OF_SIM = int(sys.argv[2])
	FOLDER = sys.argv[3]
else:
	FPS = 60
	SPEED_OF_SIM = 15
	FOLDER = input('Enter test folder name:')

#generate velocity and frame information
def vf(df, st, et, sx, sy, ex, ey, scaling = SPEED_OF_SIM, fps = FPS):
	frames = (df[et] - df[st]) / scaling * fps
	frames[frames == 0] = 0.0001
	frames = frames.values
	direction = (df[[ex, ey]].values - df[[sx, sy]].values).astype('float')
	velocity = (1/frames * direction.T).T
	return velocity, frames

#modify and create driver animation objects
def generate_animation_objects(driver_generated_points):
	#calculating velocity and frames for each movement
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

	return driver_animations

DRIVER_MOVEMENT_FILENAME = f'../{FOLDER}/driver_histories_parquet'

xy_pixel_polygons, zone_dict = create_or_load_polygon_info()

driver_generated_points = generate_positions(DRIVER_MOVEMENT_FILENAME, zone_dict, FOLDER)

driver_animations = generate_animation_objects(driver_generated_points)

"""Draws the backround of the pygame set"""
def draw_bg(poly_list, s):
	for p in poly_list:
		pygame.draw.polygon(s, 0, p[0], width = 1)

"""Loads an image with a set alpha value"""
def load_image(image_path, size = (3,3), alpha = 150):
	im = pygame.image.load(image_path)
	im = pygame.transform.scale(im, size).convert_alpha()
	im.set_alpha(alpha)
	return im
		
#initalize screen and stuff
pygame.init()
size = width, height = SCREEN_SIZE
screen = pygame.display.set_mode(size)

driver_with_passenger = load_image('passenger.png')
driver_without_passenger = load_image('nopassenger.png')
driver_idle = load_image('idle.png')

driver_rectangles = [driver_without_passenger.get_rect() for i in range(len(driver_animations))]

"""Bottom Layer with all the city boundaries drawn on"""
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
	"""Updates the position of each driver according to the current animation"""
	for driver, rect in zip(driver_animations, driver_rectangles):
		rect.center = driver.update()
		if driver.time > sys_time:
			sys_time = driver.time
		if driver.state() == 1:
			screen.blit(driver_with_passenger, rect)
		elif driver.state() == 2:
			screen.blit(driver_without_passenger, rect)
		else:
			screen.blit(driver_idle, rect)

	"""Display a system time counter"""
	time_font = font.render('Time: {hour:02}:{minute:02}'.format(hour = round(sys_time//60), minute = round(sys_time%60)), 1, (0,0,0))
	time_rect = time_font.get_rect()
	time_rect.center = (60,40)
	screen.blit(time_font, time_rect)

	pygame.display.flip()