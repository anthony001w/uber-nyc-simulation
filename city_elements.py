from collections import deque
import numpy as np
import pandas as pd
import time

class Driver:
    
    #can add behaviors here like time schedule, max distance allowed
    def __init__(self, start_zone):
        self.zone = start_zone
        self.passenger = None
        self.passenger_queue = deque()
        self.movement_history = [[0, start_zone, self.passenger]]
    
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
    
    def add_movement(self, start_time, destination, passenger = None):
        self.movement_history.append([start_time, destination, passenger])
    
    def status(self):
        return 'In Transit' if self.passenger is not None or len(self.passenger_queue) > 0 else 'Idle'
    
    def hit_maximum_passenger_queue_size(self, max_size = 4):
        return len(self.passenger_queue) >= max_size
    
    def __str__(self):
        if self.destination is not None:
            return f'In Transit to {self.destination} from {self.location} with {len(self.passenger_queue)} waiting passengers'
        else:
            return f'Currently at {self.location}'
    
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
        for driver in self.drivers:
            driver.zone = self.zone
        self.incoming_drivers = set()
        
    def get_best_available_driver(self):
        available_incoming = [d for d in self.incoming_drivers if not d.hit_maximum_passenger_queue_size()]
        if len(self.drivers) == 0:
            if len(available_incoming) == 0:
                return None
            return min(available_incoming, key = lambda d: len(d.passenger_queue))
        return min(self.drivers, key = lambda d: len(d.passenger_queue))
    
    def get_any_driver(self):
        if len(self.drivers) == 0:
            if len(self.incoming_drivers) == 0:
                return None
            return min(self.incoming_drivers, key = lambda d: len(d.passenger_queue))
        return min(self.drivers, key = lambda d: len(d.passenger_queue))
    
    def add_driver(self, driver, incoming = False):
        if incoming:
            self.incoming_drivers.add(driver)
            driver.zone = -1 * self.zone
        else:
            self.drivers.add(driver)
            driver.zone = self.zone
        
    def remove_driver(self, driver, incoming = False):
        if incoming:
            self.incoming_drivers.remove(driver)
            driver.zone = None
        else:
            self.drivers.remove(driver)
            driver.zone = None
        
    #could add additional functionality like calculating distance between zones