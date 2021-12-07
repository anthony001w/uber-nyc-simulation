from city_elements import *
from event_list import *
import numpy as np
import time

class City:
    
    def __init__(self, name, zone_list, drivers, odmatrix):
        self.name = name
        self.zones = {z.zone: z for z in zone_list}
        self.free_drivers = set(drivers)
        self.busy_drivers = set()
        self.odmatrix = []

        #convert odmatrix into a list of lists (faster access)
        for i in np.arange(1,264):
            list_of_info = [v for v in odmatrix.loc[i].values]
            self.odmatrix.append(list_of_info)
        
        #use the odmatrix to judge the closest zones
        #dictionary of values with key = zone_id
        #and the value is a pandas index listing the closest zones by mean travel time
        self.closest_zones = {}
        for i in np.arange(1,264):
            dotimes = odmatrix.loc[i]
            ordered = dotimes[~(dotimes == 0).all(axis=1)].sort_values(by = 'mean')
            if i in ordered.index:
                ordered = ordered.drop(index = i)
            self.closest_zones[i] = ordered.index
        
        #set some default value using the overall mean
        #doesn't take into account anything, is definitely a bad solution
        #better is to take into account geographic distance and maybe traffic
        default_means = []
        for i in np.arange(1,264):
            do_info = odmatrix.loc[(slice(None),i),:]
            if do_info['count'].sum() == 0:
                do_info = odmatrix.loc[(i,slice(None)),:]
            if do_info['count'].sum() != 0:
                #ignore the zone bc there's no pickups or dropoffs from it
                exp_mean = np.sum(do_info['mean'] * do_info['count']) / do_info['count'].sum()
                default_means.append(exp_mean)
            else:
                default_means.append(np.mean(default_means))
        self.default_times = default_means

        self.timed_stats = {'generating_movement_times':[0,0],
                            'choose_driver':[0,0]}
        
    def get_zone(self, zone_id):
        return self.zones[zone_id]

    def process_event(self, event):

        if event.type == 'Arrival':
            return self.process_arrival_event(event)

        elif event.type == 'Movement':
            return self.process_movement_event(event)

        elif event.type == 'Trip':
            return self.process_trip_event(event)
    
    def generate_movement_time(self, pu, do):
        
        tic = time.time()
        movement_info = self.odmatrix[pu - 1][do - 1]
        if (movement_info == 0).all():
            #if there's no movement information, try to generate an exponential var from the weighted
            #mean for the dropoff location
            m = np.random.exponential(self.default_times[do - 1])
        else:
            m = max(np.random.normal(loc = movement_info[0], scale = movement_info[1]), movement_info[2])
        toc = time.time()
        self.timed_stats['generating_movement_times'][0] += toc - tic
        self.timed_stats['generating_movement_times'][1] += 1
        return m
    
    def process_arrival_event(self, event):
        pickup_zone = self.get_zone(event.passenger.start)
        dropoff_zone = self.get_zone(event.passenger.end)
        
        #get best available driver and check status
        chosen_driver = pickup_zone.get_available_driver()
        if chosen_driver is not None:
            if not chosen_driver.is_moving():
                #return a trip event
                #update pickup zone drivers and dropoff zone drivers
                #edit driver's passenger variable
                #update driver movement history
                pickup_zone.remove_driver(chosen_driver)
                chosen_driver.passenger = event.passenger
                chosen_driver.add_start_of_movement(event.time, event.passenger.start)
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
            chosen_driver = None
            
            tic = time.time()
            #pick a random free driver
            if len(self.free_drivers) > 0:
                for zone_id in self.closest_zones[event.passenger.start]:
                    zone = self.get_zone(zone_id)
                    chosen_driver = zone.get_available_driver()
                    if chosen_driver is not None:
                        break

                if chosen_driver is None:
                    chosen_driver = self.free_drivers.pop()
                    self.free_drivers.add(chosen_driver)
                    zone = self.get_zone(chosen_driver.last_location)

            #last case scenario if no driver is available, just choose any driver
            if chosen_driver is None:
                chosen_driver = self.busy_drivers.pop()
                self.busy_drivers.add(chosen_driver)
            toc = time.time()
            self.timed_stats['choose_driver'][0] += toc - tic
            self.timed_stats['choose_driver'][1] += 1
            
            #using this chosen driver, check status
            if not chosen_driver.is_moving():
                #return a movement event
                #update the zone's drivers and the passenger's pickup zone drivers
                #update the driver's movement history
                zone.remove_driver(chosen_driver)
                self.free_drivers.remove(chosen_driver)
                self.busy_drivers.add(chosen_driver)

                chosen_driver.add_start_of_movement(event.time, zone.zone)
                chosen_driver.add_passenger(event.passenger)
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
        
        driver.add_end_of_movement(event.time, event.end_zone.zone)
        driver.add_start_of_movement(event.time, event.end_zone.zone)
        driver.passenger = passenger
        
        return Movement(event.time, driver, dropoff_zone, passenger.service, passenger)
    
    def process_trip_event(self, event):
        
        #end of a trip event, driver will be dropping off a passenger and will need to get the next passenger in queue
        #or idle if no other passenger
        current_passenger = event.passenger
        current_passenger.departure_time = event.time
        driver = event.driver
        driver.passenger = None
        driver.add_end_of_movement(event.time, current_passenger.end, current_passenger)
        
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
                
                driver.passenger = passenger
                driver.add_start_of_movement(event.time, passenger.start)
                
                return Movement(event.time, driver, zone, passenger.service, passenger)
            
            else:
                #generate a movement event to the next passenger
                zone = self.get_zone(passenger.start)
                driver.add_start_of_movement(event.time, current_passenger.end)

                return Movement(event.time, driver, zone, self.generate_movement_time(event.end_zone.zone, zone.zone))

    def formatted_stats(self):
        s = ''
        for name in self.timed_stats:
            s += f'\n\t-- {name} --\n\tTotal Time Spent: {self.timed_stats[name][0]}\n\t# of Occurences: {self.timed_stats[name][1]}'
        return s
                
        