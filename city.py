from city_elements import *
from event_list import *
import pandas as pd
import numpy as np
import time

class City:
    
    def __init__(self, name, zone_list, drivers, odmatrix):
        self.name = name
        self.zones = {z.zone: z for z in zone_list}
        self.free_drivers = set(drivers)
        self.busy_drivers = set()
        self.odmatrix = odmatrix
        
        #use the odmatrix to judge the closest zones
        #dictionary of values with key = zone_id
        #and the value is a Pandas series where the index is the zone id and the value is the mean trip time
        #ordered ascending in terms of mean trip time (not including the same zone)
        #used for choosing drivers
        self.closest_zones = {}
        for i in odmatrix.index:
            ordered = odmatrix.loc[i].dropna().sort_values()
            if i in ordered.index:
                ordered = ordered.drop(index = i)
            self.closest_zones[i] = ordered
        
        #set some default value using the overall mean
        #doesn't take into account anything, is definitely a bad solution
        #better is to take into account geographic distance and maybe traffic
        self.default_movement_mean = np.nanmean(odmatrix.values)
        
    def get_zone(self, zone_id):
        return self.zones[zone_id]
    
    #generates a random movement time assuming service times are exponentially distributed
    def generate_movement_time(self, pu, do):

        #try except just catches the case where there's no pickup data for a zone
        try:
            mm = self.odmatrix.loc[pu, do]
            if not pd.isnull(mm):
                mt = np.random.exponential(mm)
            else:
                #if there's no data for that specific pu->do, just use the mean of the pu zone trips
                mt = np.random.exponential(self.odmatrix.loc[pu].mean())
        except:
            #if no data on that pickup zone, use the mean of the dropoff zone
            mt = np.random.exponential(self.odmatrix.loc[:,do].mean())
        return mt
            
    
    def process_arrival_event(self, event):
        pickup_zone = self.get_zone(event.passenger.start)
        dropoff_zone = self.get_zone(event.passenger.end)
        
        #get best avaiable driver and check status
        chosen_driver = pickup_zone.get_best_available_driver()
        if chosen_driver is not None:
            if chosen_driver.status() == 'Idle':
                #return a trip event
                #update pickup zone drivers and dropoff zone drivers
                #edit driver's passenger variable
                #update driver movement history
                pickup_zone.remove_driver(chosen_driver)
                dropoff_zone.add_driver(chosen_driver, incoming = True)
                chosen_driver.passenger = event.passenger
                chosen_driver.add_movement(event.time, event.passenger.end, event.passenger)
                self.free_drivers.remove(chosen_driver)
                self.busy_drivers.add(chosen_driver)
                return Movement(event.time, chosen_driver, dropoff_zone, event.passenger.service, event.passenger)
            else:
                #just add the passenger to the driver's queue
                #return nothing
                chosen_driver.add_passenger(event.passenger)
        else:
            
            #pick another driver based on the closest zones
            #if there's no free driver in the list of closest zones, pick a free driver out of the whole list
            for zone_id in self.closest_zones[event.passenger.start].index:
                zone = self.get_zone(zone_id)
                if len(zone.drivers) != 0:
                    chosen_driver = zone.get_best_available_driver()
                    break
            
            #pick a random free driver
            if len(self.free_drivers) > 0:
                chosen_driver = self.free_drivers.pop()
                self.free_drivers.add(chosen_driver)
                zone = self.get_zone(chosen_driver.zone)

            #last case scenario if no driver is available, just choose any driver
            if chosen_driver is None:
                chosen_driver = self.busy_drivers.pop()
                self.busy_drivers.add(chosen_driver)
            
            #using this chosen driver, check status
            if chosen_driver.status() == 'Idle':
                #return a movement event
                #update the zone's drivers and the passenger's pickup zone drivers
                #update the driver's movement history
                zone.remove_driver(chosen_driver)
                pickup_zone.add_driver(chosen_driver, incoming = True)
                chosen_driver.add_movement(event.time, event.passenger.start)
                chosen_driver.add_passenger(event.passenger)
                self.free_drivers.remove(chosen_driver)
                self.busy_drivers.add(chosen_driver)
                #generate a movement time from zone to zone
                return Movement(event.time, chosen_driver, pickup_zone, self.generate_movement_time(zone.zone, pickup_zone.zone))
            
            else:
                #add the passenger to the driver's queue
                chosen_driver.add_passenger(event.passenger)
    
    def process_movement_event(self, event):
        
        #at the end of a movement event, the driver will be picking up a passenger in queue
        #pop the next passenger
        #change the driver's location in zone data
        #update driver movement history
        #update driver passenger
        #return another movement event (which is a trip event)
        driver = event.driver
        passenger = driver.pop_next_passenger()
        
        pickup_zone = event.end_zone
        dropoff_zone = self.get_zone(passenger.end)
        
        pickup_zone.remove_driver(driver, incoming = True)
        dropoff_zone.add_driver(driver, incoming = True)
        
        driver.add_movement(event.time, passenger.start)
        driver.add_movement(event.time, passenger.end, passenger)
        driver.passenger = passenger
        
        return Movement(event.time, driver, dropoff_zone, passenger.service, passenger)
    
    def process_trip_event(self, event):
        
        #end of a trip event, driver will be dropping off a passenger and will need to get the next passenger in queue
        #or idle if no other passenger
        current_passenger = event.passenger
        current_passenger.departure_time = event.time
        driver = event.driver
        driver.passenger = None
        driver.add_movement(event.time, current_passenger.end)
        event.end_zone.remove_driver(driver, incoming = True)
        
        #get next passenger
        passenger = driver.get_next_passenger()
        if passenger is None:
            event.end_zone.add_driver(driver)
            self.free_drivers.add(driver)
            self.busy_drivers.remove(driver)
        else:
            #2 cases, either the passenger is in the zone (generating a trip event)
            #or the passenger is in another zone (generating a movement event to that zone)
            if current_passenger.end == passenger.start:
                #generate a trip event
                #change driver location
                #update driver movement
                #update driver passenger
                #return a trip event
                passenger = driver.pop_next_passenger()
                zone = self.get_zone(passenger.end)
                zone.add_driver(driver, incoming = True)
                driver.add_movement(event.time, passenger.end, passenger)
                driver.passenger = passenger
                
                return Movement(event.time, driver, zone, passenger.service, passenger)
            
            else:
                #generate a movement event to the next passenger
                zone = self.get_zone(passenger.start)
                zone.add_driver(driver, incoming = True)
                driver.add_movement(event.time, passenger.start)
                
                return Movement(event.time, driver, zone, self.generate_movement_time(event.end_zone.zone, zone.zone))
                
        