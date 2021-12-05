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
if len(sys.argv) == 5:
	FPS = int(sys.argv[1])
	SPEED_OF_SIM = int(sys.argv[2])
	FOLDER = sys.argv[3]
	MODE = sys.argv[4]
else:
	FOLDER = input('Enter test folder name:')
	FPS = int(input('Enter FPS: '))
	SPEED_OF_SIM = int(input('Enter the speed of the simulation: '))
	MODE = input('Mode (random/lines): ')

DRIVER_MOVEMENT_FILENAME = f'../{FOLDER}/driver_histories_parquet'

xy_pixel_polygons, zone_dict = create_or_load_polygon_info()

driver_generated_points = generate_positions(DRIVER_MOVEMENT_FILENAME, zone_dict, FOLDER, mode = MODE)

driver_animations = DriverAnimation(driver_generated_points, SPEED_OF_SIM, FPS)

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

driver_rectangles = [driver_without_passenger.get_rect() for i in range(driver_animations.driver_count)]

"""Bottom Layer with all the city boundaries drawn on"""
layer1 = pygame.Surface(SCREEN_SIZE)
layer1.fill([225,225,225])
draw_bg(xy_pixel_polygons, layer1)

clock = pygame.time.Clock()
font = pygame.font.SysFont('Trebuchet MS', 20)
while True:
	clock.tick(FPS)

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			quit()

	screen.blit(layer1, (0,0))
	"""Updates the position of each driver according to the current animation"""
	centers, is_moving, has_pass, finished, curr_time = driver_animations.update()
	for c, i, h, f, rect in zip(centers, is_moving, has_pass, finished, driver_rectangles):
		rect.center = c
		if f:
			screen.blit(driver_idle, rect)
		elif h:
			screen.blit(driver_with_passenger, rect)
		elif i:
			screen.blit(driver_without_passenger, rect)
		else:
			screen.blit(driver_idle, rect)

	"""Display a system time counter"""
	time_font = font.render('Time: {hour:02}:{minute:02}'.format(hour = round(curr_time//60), minute = round(curr_time%60)), 1, (0,0,0))
	time_rect = time_font.get_rect()
	time_rect.center = (60,40)
	screen.blit(time_font, time_rect)

	pygame.display.flip()