from city_elements import *
from event_list import *
import numpy as np
import time

class ZoneDict:
    """Representing every zone in a dictionary of sets with keys as the zone ids"""

    def __init__(self, zone_ids):
        self.zones = {z:set() for z in zone_ids}

    def add_driver(self, zone_id, driver):
        self.zones[zone_id].add(driver)
    
    def remove_driver(self, zone_id, driver):
        self.zones[zone_id].remove(driver)

    def get_driver(self, zone_id):
        if len(self.zones[zone_id]) == 0:
            return None
        else:
            d = self.zones[zone_id].pop()
            self.zones[zone_id].add(d)
            return d

    def get_driver_from_any_zone(self, zones_to_check):
        for z in zones_to_check:
            driver = self.get_driver(z)
            if driver:
                return driver
        return None
    
    def shift_driver(self, driver, old_zone, new_zone):
        self.remove_driver(old_zone, driver)
        self.add_driver(new_zone, driver)

    def initialize(self, driver_list):
        for d in driver_list:
            if d.end < d.start:
                self.add_driver(d.start_zone, d)
                    
    def get_driver_counts_in_system(self):
        return {i:len(self.zones[i]) for i in self.zones}

class DriverStatus:
    """Representing driver statuses as a whole in a dictionary of sets where the names are the statuses"""
    def __init__(self, priority_names):
        self.status = {p:set() for p in priority_names}

    def shift_driver(self, driver, old_status, new_status):
        self.status[old_status].remove(driver)
        self.status[new_status].add(driver)

    def driver_in_status(self, driver, status):
        return driver in self.status[status]

    def get_driver_from_status(self, status):
        if len(self.status[status]) == 0:
            return None
        else:
            d = self.status[status].pop()
            self.status[status].add(d)
            return d

    def add_driver(self, driver, status):
        self.status[status].add(driver)

    def get_status_counts(self):
        return {s:len(self.status[s]) for s in self.status}

class City:
    
    def __init__(self, name, zone_ids, drivers, odmatrix):
        self.name = name
        self.zones = ZoneDict(zone_ids)
        self.unserved_customers = deque()
        self.driver_status = DriverStatus(['inactive','free','busy','max_queue','marked_for_departure'])

        #add available drivers to the set of free drivers based on end < start
        for d in drivers:
            if d.end < d.start:
                self.driver_status.add_driver(d, 'free')
            else:
                self.driver_status.add_driver(d, 'inactive')
        self.zones.initialize(drivers)

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

        self.timed_stats = {'generating_movement_times':[0,0]}

    def process_event(self, event):

        if event.type == 'Arrival':
            return self.process_arrival_event(event)

        elif event.type == 'Movement':
            return self.process_movement_event(event)

        elif event.type == 'Trip':
            return self.process_trip_event(event)

        elif event.type == 'Driver Arrival':
            return self.process_driver_arrival(event)

        elif event.type == 'Driver Departure':
            return self.process_driver_departure(event)
    
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
        pickup_zone = event.passenger.start

        #first search for a driver in the same pickup zone as the passenger
        chosen_driver = self.zones.get_driver(pickup_zone)
        if chosen_driver:
            #if the driver is not moving, then the driver can immediately serve the passenger
            #system changes - remove driver from the pickup zone and shift the driver's status
            self.zones.remove_driver(pickup_zone, chosen_driver)
            self.driver_status.shift_driver(chosen_driver, 'free', 'busy')

            #add the passenger to the driver queue and send the driver to the same zone
            chosen_driver.add_passenger(event.passenger)
            chosen_driver.add_start_of_movement(event.time, pickup_zone)
            movement_time = self.generate_movement_time(pickup_zone, pickup_zone)

            #generate a movement event
            return Movement(event.time + movement_time, chosen_driver, pickup_zone, pickup_zone)
        else:
            chosen_driver = None
            status_counts = self.driver_status.get_status_counts()
            
            #choosing a driver
            if status_counts['free'] > 0:
                #only look at the 5 closest zones
                some_close_zones = self.closest_zones[pickup_zone][:5]
                chosen_driver = self.zones.get_driver_from_any_zone(some_close_zones)

                if chosen_driver is None:
                    chosen_driver = self.driver_status.get_driver_from_status('free')
                zone = chosen_driver.last_location
                
                #the chosen driver is free, so generate a movement event from the driver to the customer
                self.zones.remove_driver(zone, chosen_driver)
                self.driver_status.shift_driver(chosen_driver, 'free', 'busy')

                #add passenger and add the start of a movement
                chosen_driver.add_passenger(event.passenger)
                chosen_driver.add_start_of_movement(event.time, zone)

                #generate a movement time for the movement
                movement_time = self.generate_movement_time(zone, pickup_zone)
                return Movement(event.time + movement_time, chosen_driver, zone, pickup_zone)

            elif status_counts['busy'] > 0:
                #choose any busy driver
                chosen_driver = self.driver_status.get_driver_from_status('busy')

                #add passenger to the driver's queue
                chosen_driver.add_passenger(event.passenger)

                #shift the driver's status to max queue if the max queue is hit
                if chosen_driver.hit_max_queue():
                    self.driver_status.shift_driver(chosen_driver, 'busy','max_queue')

            else:
                self.unserved_customers.append(event.passenger)
    
    def process_movement_event(self, event):

        driver = event.driver
        passenger = driver.pop_next_passenger()

        #if the driver's queue length is no longer the max queue, shift the driver's status
        if self.driver_status.driver_in_status(driver, 'max_queue') and not driver.hit_max_queue():
            self.driver_status.shift_driver(driver, 'max_queue', 'busy')
        
        driver.add_end_of_movement(event.time, event.end_zone)
        driver.add_start_of_movement(event.time, event.end_zone)
        driver.passenger = passenger

        return Trip(event.time + passenger.service, driver, passenger)
    
    def process_trip_event(self, event):
        
        #at the end of a trip event the driver goes to the next passenger, is idle, or picks up from the unserved queue
        current_passenger = event.passenger
        current_passenger.departure_time = event.time
        driver = event.driver
        driver.passenger = None
        driver.add_end_of_movement(event.time, current_passenger.end, current_passenger)
        
        #get next passenger
        passenger = driver.get_next_passenger()
        if passenger is None:
            self.zones.add_driver(event.end_zone(), driver)
            if self.driver_status.driver_in_status(driver, 'marked_for_departure'):
                self.driver_status.shift_driver(driver, 'marked_for_departure', 'free')
            else:
                self.driver_status.shift_driver(driver, 'busy', 'free')

            if driver.out_of_schedule(event.time):
                driver_dep_event = DriverDeparture(driver, event.time)
                return driver_dep_event
            else:
                #if no departure is scheduled, start serving the customers waiting
                if len(self.unserved_customers) > 0:
                    passenger = self.unserved_customers.popleft()
                    return self.serve_unserved_passenger(event.time, current_passenger.end, driver, passenger)

        else:
            #move to the next passenger
            driver.add_start_of_movement(event.time, current_passenger.end)
            movement_time = self.generate_movement_time(current_passenger.end, passenger.start)
            return Movement(event.time + movement_time, driver, current_passenger.end, passenger.start)

    def process_driver_arrival(self, event):
        #add the driver to the free driver pool
        #add the driver to the zone he starts in
        #remove from inactive driver list
        driver = event.driver
        driver.last_location = driver.start_zone
        self.driver_status.shift_driver(driver, 'inactive', 'free')
        self.zones.add_driver(driver.start_zone, driver)

        if len(self.unserved_customers) > 0:
            passenger = self.unserved_customers.popleft()
            return self.serve_unserved_passenger(event.time, driver.start_zone, driver, passenger)

    def process_driver_departure(self, event):

        driver = event.driver

        #if the driver is busy, just do nothing and another departure event will be generated
        if self.driver_status.driver_in_status(driver, 'free'):
            self.driver_status.shift_driver(driver, 'free', 'inactive')
            self.zones.remove_driver(driver.last_location, driver)
        #shift the other drivers into marked for departure, so that they aren't given new passengers
        elif self.driver_status.driver_in_status(driver, 'busy'):
            self.driver_status.shift_driver(driver, 'busy', 'marked_for_departure')
        elif self.driver_status.driver_in_status(driver, 'max_queue'):
            self.driver_status.shift_driver(driver, 'max_queue', 'marked_for_departure')

    def serve_unserved_passenger(self, current_time, current_location, driver, passenger):
        #shift the driver from free to busy and remove the driver from where they are currently
        self.driver_status.shift_driver(driver, 'free', 'busy')
        self.zones.remove_driver(driver.last_location, driver)

        #add passenger and movement history
        driver.add_passenger(passenger)
        driver.add_start_of_movement(current_time, current_location)

        movement_time = self.generate_movement_time(current_location, passenger.start)
        return Movement(current_time + movement_time, driver, current_location, passenger.start)

    def formatted_stats(self):
        s = ''
        for name in self.timed_stats:
            s += f'\n\t-- {name} --\n\tTotal Time Spent: {self.timed_stats[name][0]}\n\t# of Occurences: {self.timed_stats[name][1]}'
        return s
                
        