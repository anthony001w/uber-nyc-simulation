from city_elements import *
import time

class Event:
    
    def __init__(self, time_of_event):  
        self.time = time_of_event
        
class Arrival(Event):
    
    def __init__(self, passenger):
        Event.__init__(self, passenger.time)
        self.passenger = passenger
        self.type = 'Arrival'
        
    def __str__(self):
        return 'Arrival\n' + str(self.passenger)

class DriverArrival(Event):
    
    def __init__(self, driver):
        Event.__init__(self, driver.start)
        self.driver = driver 
        self.type = 'Driver Arrival'
    
class DriverDeparture(Event):
    
    def __init__(self, driver, t = 0):
        t = max(t, driver.end)
        Event.__init__(self, t)
        self.driver = driver 
        self.type = 'Driver Departure'
    
class Movement(Event):
        
    def __init__(self, current_time, driver, destination_zone, driving_time, passenger = None):
        Event.__init__(self, current_time + driving_time)
        self.driving_time = driving_time
        self.driver = driver
        self.passenger = passenger
        self.end_zone = destination_zone
        self.type = 'Movement' if passenger is None else 'Trip'
    
    def __str__(self):
        return f'Movement\nEnd Time: {self.time} \nEnd Zone: {self.end_zone} \nCurrent Time: {self.time - self.driving_time}'

class EventList:
    
    def __init__(self, initial_event_list):
        self.events = deque(sorted(initial_event_list, key = lambda e: e.time))
        self.timed_stats = {'insertion speed':[0,0], 
                            'pop speed':[0,0], 
                            'search speed':[0,0]}
        
    def insert_event(self, event):
        
        #binary search
        tic = time.time()
        bounds = [0,len(self.events) - 1]
        while bounds[0] < bounds[1]:
            i = (bounds[1] + bounds[0])//2
            if self.events[i].time > event.time:
                bounds[1] = i - 1
            elif self.events[i].time == event.time:
                bounds[0] = i
                bounds[1] = i
            else:
                bounds[0] = i + 1
        toc = time.time()
        self.timed_stats['search speed'][0] += toc - tic
        self.timed_stats['search speed'][1] += 1
        
        tic = time.time()
        self.events.insert(bounds[0], event)
        toc = time.time()
        self.timed_stats['insertion speed'][0] += toc - tic
        self.timed_stats['insertion speed'][1] += 1
        
    def iterate_next_event(self):
        tic = time.time()
        e = self.events.popleft()
        toc = time.time()
        self.timed_stats['pop speed'][0] += toc - tic
        self.timed_stats['pop speed'][1] += 1
        return e
        
    def is_finished(self):
        return len(self.events) == 0

    def formatted_stats(self):
        s = ''
        for name in self.timed_stats:
            s += f'\n\t-- {name} --\n\tTotal Time Spent: {self.timed_stats[name][0]}\n\t# of Occurences: {self.timed_stats[name][1]}'
        return s