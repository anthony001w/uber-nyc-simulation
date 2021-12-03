from collections import deque
import numpy as np
import pandas as pd
import time

class Driver:
    
    #can add behaviors here like time schedule, max distance allowed
    def __init__(self, start_zone):
        self.last_location = start_zone
        self.last_time = 0
        self.passenger = None
        self.passenger_queue = deque()
        self.movement_history = []
    
    def add_passenger(self, passenger):
        self.passenger_queue.append(passenger)
        
    def pop_next_passenger(self):
        if len(self.passenger_queue) == 0:
            return None
        return self.passenger_queue.popleft()
    
    def get_next_passenger(self):
        if len(self.passenger_queue) == 0:
            return None
        return self.passenger_queue[0]
    
    def add_start_of_movement(self, start_time, start):
        #start time, end time, last location, next location, is moving, has passenger
        self.movement_history.append((self.last_time, start_time, self.last_location, start, False, False))
        self.last_time = start_time

    def add_end_of_movement(self, end_time, end, passenger = None):
        self.movement_history.append((self.last_time, end_time, self.last_location, end, True, passenger != None))
        self.last_time = end_time
        self.last_location = end
    
    def status(self):
        return 'In Transit' if self.passenger is not None or len(self.passenger_queue) > 0 else 'Idle'

    def return_movement_dataframe(self):
        return pd.DataFrame(self.movement_history, columns = ['start_time', 'end_time', 'start_zone', 'end_zone', 'is_moving', 'has_passenger'])
    
class Passenger:
    
    def __init__(self, arrival_time, start_zone, end_zone, service_time):
        self.time = arrival_time
        self.start = int(start_zone)
        self.end = int(end_zone)
        self.departure_time = None
        self.service = service_time
        
    def waiting_time(self):
        
        return self.departure_time - self.service - self.time
        
    def __str__(self):
        return f"Arrival Time: {self.time} \nStart Zone: {self.start} \nEnd Zone: {self.end} \nTrip Time: {self.service}"
    
class Zone:
    
    #stores drivers
    def __init__(self, zone_id, driver_set):
        self.zone = zone_id
        self.drivers = driver_set
        
    def get_available_driver(self):
        if len(self.drivers) == 0:
            return None
        d = self.drivers.pop()
        self.drivers.add(d)
        return d
    
    def add_driver(self, driver):
        self.drivers.add(driver)
        
    def remove_driver(self, driver):
        self.drivers.remove(driver)
        
    #could add additional functionality like calculating distance between zones