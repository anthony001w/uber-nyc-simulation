import pandas as pd
from sodapy import Socrata
import sys, pygame, os
import numpy as np
from joblib import dump,load

SCREEN_SIZE = (1200,600)

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
                 np.floor(bbox_dim[1] * np.abs(top - merc[:,1]) / total_dim[1])]

def extract_pixel_positions_of_coordinates(taxi_zone_information):
	#extract all the coordinate points first (taking into account weird shapes)
	polygons = []
	for i,z in taxi_zone_information.iterrows():
	    p = z['the_geom']['coordinates']
	    
	    add_polygons(p, polygons, z.location_id)
	    
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
		xy_pixel_polygons.append(convert_coords(p, mercs[left][0], mercs[top][1], (d_x, d_y)))

	return xy_pixel_polygons

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

	xy_pixel_polygons = extract_pixel_positions_of_coordinates(taxi_zones)

	dump(xy_pixel_polygons, 'polygon_info')
else:
	xy_pixel_polygons = load('polygon_info')

#initalize screen and stuff
size = width, height = SCREEN_SIZE

screen = pygame.display.set_mode(size)

#initial fill in
drawn_pgons = []
screen.fill([160, 160, 160])
for pgon in xy_pixel_polygons:
	#draw black lines with a width of 1
	drawn_pgons.append(pygame.draw.polygon(screen, 0, pgon, width = 1))

while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			quit()

	#choose a random box to highlight
	# d = np.random.choice(np.arange(0,len(drawn_pgons)))
	# pygame.draw.polygon(screen, 128, xy_pixel_polygons[d], width = 0)

	#reset the rest of the boxes
	pygame.display.flip()