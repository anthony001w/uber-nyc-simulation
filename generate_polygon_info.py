import pandas as pd
from sodapy import Socrata
import os
import numpy as np
from joblib import dump,load
from shapely.geometry import Polygon
from tqdm import tqdm
from city_elements import *
from city import *
from event_list import *
from matplotlib.path import Path

SCREEN_SIZE = (1200,800)

"""Adds polygons recursively from lists"""
def add_polygons(lists, region_id):
	plist = []
	if len(lists) > 1 and np.array([len(l) == 2 for l in lists]).all():
		return (lists, region_id)
	else:
		for l in lists:
			plist.extend(add_polygons(l, region_id))
		return plist

"""Converts latitude and longitude arrays to XY coordinates using Mercator Projection"""
def merc_from_arrays(lats, lons):
    r_major = 6378137.000
    x = r_major * np.radians(lons)
    scale = x/lons
    y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + lats * (np.pi/180.0)/2.0)) * scale
    return np.c_[x, y]

"""converts (lon,lat) from socrata -> pixel positions of the bounding box"""
def convert_coords(polygon_coords, left, top, total_dim, bbox_dim = SCREEN_SIZE):

    p = np.array(polygon_coords[0])
    merc = merc_from_arrays(p[:,1], p[:,0])

    bbox_coords = np.c_[np.floor(bbox_dim[0] * np.abs(merc[:,0] - left) / total_dim[0]), 
                  	np.floor(bbox_dim[1] * np.abs(top - merc[:,1]) / total_dim[1])]

    return bbox_coords, polygon_coords[1]

"""Uses previous 3 functions to extract zone coordinates as polygons from the Socrata dataset"""
def extract_zones_as_polygons(taxi_zone_information):
	#extract all the coordinate points first (taking into account weird shapes)
	polygons = []
	for i,z in taxi_zone_information.iterrows():
		p = z['the_geom']['coordinates']
		polygons.extend(add_polygons(p, int(z.objectid)))
	    
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

"""Converts polygons into a zone:best_polygon dictionary for sampling random points from"""
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


def create_or_load_polygon_info(file_name = 'viz/polygon_info'):
	#polygon data only needs to be created once
	if not os.path.exists(file_name):
		with Socrata('data.cityofnewyork.us', 'vA3MfkSw5kKhpzNkitJkv5yFP') as client:
			#gets the taxi zones (the geometry of each zone in the city + the zone name and borough)
			taxi_zones =  pd.DataFrame.from_records(client.get('755u-8jsi'))

		xy_pixel_polygons = extract_zones_as_polygons(taxi_zones)
		zone_dict = convert_polygons_for_sampling(xy_pixel_polygons)

		dump([xy_pixel_polygons, zone_dict], file_name)

		return xy_pixel_polygons, zone_dict
	else:
		polygon_info = load(file_name)
		return polygon_info[0], polygon_info[1]