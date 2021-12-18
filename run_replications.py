import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import load
from city_elements import *
from city import *
from event_list import *
import os
import sys

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a+")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    

pickup_data = pd.read_pickle('input_data/arrival_and_dropoff_distributions')
hourly_arrival_rate =  pickup_data.apply(lambda item: item[0])
dropoff_frequency  = pickup_data.apply(lambda  item: item[1] / item[1].sum())
trip_time_data = pd.read_parquet('input_data/trip_time_means')

def generate_bundle(size = 1000):
    """Generates uniform centers of intervals (each interval represents a driver schedule
       Generates intervals of lengths centered around 8 hours with min/max of 2/10
    """
    centers = np.random.randint(1440, size = size)

    #weights 8 hour intervals higher than 2 hours or 10 hours
    possible_lengths = np.arange(60, 301) * 2
    length_pvalues = 1/(np.abs(480 - possible_lengths) + 15)
    length_pvalues = length_pvalues / length_pvalues.sum()
    lengths = np.random.choice(possible_lengths, size = size, p = length_pvalues)
    lengths.sort()
    #the sort is so that when we iteratively add them to the driver list we add the highest values first
    return centers, lengths[::-1]

def update_availability_arr(availability, bundle, preferred_availability, acceptable_overlap = 60, show_progress = False):
    """Updates the available drivers using the bundle of given intervals"""
    mins = bundle[0] - bundle[1] / 2
    mins = np.where(mins >= 0, mins, 1440 + mins)
    maxs = bundle[0] + bundle[1] / 2
    maxs = np.where(maxs < 1440, maxs, maxs - 1440)
    bounds = np.c_[mins, maxs].astype(int)

    #transform bounds to np array (1440)
    #and iteratively add if it's okay to add
    updated_avail = availability
    accepted_drivers = []
    if show_progress:
        iterable_shown = tqdm(range(len(bounds)), position = 0, leave = True, desc = 'Transforming bounds to arrays')
    else:
        iterable_shown = range(len(bounds))
    for i in iterable_shown:
        new_arr = np.zeros(1440)
        s, e = bounds[i][0], bounds[i][1]
        if e < s:
            new_arr[s:] = 1
            new_arr[:e] = 1
        else:
            new_arr[s:e] = 1
        restricted = (preferred_availability - updated_avail <= 0).astype(int)
        if (new_arr + restricted == 2).sum() <= acceptable_overlap:
            accepted_drivers.append(i)
            updated_avail += new_arr
                        
    #filter out the bounded_arr
    diff = (updated_avail - preferred_availability)
    return updated_avail, diff, bounds[accepted_drivers]

"""Tolerated_under_preferred -> maximum difference between 
the number of generated drivers at any minute and the number of preferred drivers at any minute

Acceptable_overlap -> the number of minutes that can be overlapped when adding a driver to the generated driver list

chunk_size -> number of driver schedules generated at once
"""
def generate_driver_schedules(preferred_availability, 
    tolerated_under_preferred = 3000, 
    acceptable_overlap = 60, 
    chunk_size = 100000,
    show_progress = False):
    """Given a few parameters, generate driver schedules until some acceptable threshold is met for the given
       preferred availability function

       preferred_availability = array(1440) that represents the # of preferred drivers at each minute of the day 
    """
    avail = np.zeros(1440)
    avail, diff, schedules = update_availability_arr(avail, generate_bundle(chunk_size), preferred_availability, acceptable_overlap, show_progress)
    print(f'Maximu Difference between # Drivers Available and Preferred Amount: {diff.min().round()}', end = ' ')
    while diff.min() <= -tolerated_under_preferred:
        avail, diff, s2 = update_availability_arr(avail, generate_bundle(chunk_size), preferred_availability, acceptable_overlap, show_progress)
        print(diff.min().round(), end = ' ')
        schedules = np.append(schedules, s2, axis = 0)
    print()
    return schedules

def generate_arrivals_per_zone(zone_hourly_arrivals = hourly_arrival_rate, 
                               zone_dropoff_frequencies = dropoff_frequency, 
                               zone_to_zone_times = trip_time_data, 
                               one_list = True,
                               show_progress_bar = False):
    
    #check to make sure the indices match
    assert (zone_hourly_arrivals.index == zone_dropoff_frequencies.index).all()
    
    zone_arrivals = []
    #for each zone, generate a day's worth of arrivals
    iterable = zone_hourly_arrivals.index if not show_progress_bar else tqdm(zone_hourly_arrivals.index, position = 0, leave = True, desc = 'Zone Arrivals Generated')
    for i in iterable:
        
        hourly_rates = zone_hourly_arrivals.loc[i]
        dropoff_dist = zone_dropoff_frequencies.loc[i]
        zone_service_times = zone_to_zone_times.loc[i]
                
        max_rate = hourly_rates.max()
        #rate = max_rate / 60 minutes (since max_rate is in minutes)
        #input the inverse as the mean interarrival time (scale parameter for np.random.exponential)
        temp_interarrivals = np.random.exponential(scale = 60/max_rate, size = 25000)
        while temp_interarrivals.cumsum().max() <= 24 * 60:
            temp_interarrivals = np.append(temp_interarrivals, np.random.exponential(scale = 60/max_rate, size = 25000))
        
        #this cuts off interarrivals at 1 day
        interarrivals = temp_interarrivals[temp_interarrivals.cumsum() <= 24*60]
        arrivals = interarrivals.cumsum()
                
        #thinning process
        #uses constant hourly rate (like a 24 part step function) to generate the thinning probabilities
        keep_probability = (hourly_rates[(arrivals // 60).astype(int)] / max_rate).values
        unif = np.random.uniform(size = arrivals.shape[0])
        kept_arrivals = arrivals[unif <= keep_probability]
                
        #for each arrival generate from the dropoff distribution
        dropoffs = np.random.choice(dropoff_dist.index, size = kept_arrivals.shape[0], p = dropoff_dist)
                              
        #generate data in the form of (time, dropoff location id, pickup location id)
        arrival_data = np.vstack([kept_arrivals, dropoffs, i*np.ones(kept_arrivals.shape[0])]).T
        
        #format into dataframe
        arrival_df = pd.DataFrame(data = arrival_data, columns = ['time','dolocationid','pulocationid'])
        
        #each arrival, generate a service time from the service time distributions
        #this is SLOW
        if len(arrival_df) > 0:
            services = [max(np.random.normal(loc = info[0], scale = info[1]), 0)
                        for info in zone_service_times.loc[arrival_df.dolocationid].values]

            arrival_df['service'] = services

            zone_arrivals.append(arrival_df)
    
    #if one list, then combine everything into one big arrival matrix
    #otherwise, just return the list of arrival dataframes
    if one_list:
        zone_arrivals = pd.concat(zone_arrivals).sort_values('time').reset_index(drop=True)
    
    return zone_arrivals

def simulate_with_individual_drivers(arrivals,
                                     preferred_driver_availability,
                                     driver_distribution = 'proportional',
                                     odmatrix = trip_time_data,
                                     pickup_data = hourly_arrival_rate):
    #convert arrivals into passengers, and then into events
    passengers = []
    drivers = []
    
    initial_events = deque()
    for a in tqdm(arrivals.values, position = 0, leave = True, desc = 'Passenger Objects Created'):
        p = Passenger(a[0], a[1], a[2], a[3])
        passengers.append(p)
        initial_events.append(Arrival(p))
    
    #setup drivers and zones based on driver_distribution parameter
    #setup driver schedules and insert driver arrivals and departures into the initial event list 
    #everything is under the city class
    if driver_distribution == 'proportional':

        #generate driver schedules
        dschedules = generate_driver_schedules(preferred_driver_availability)
        driver_count = len(dschedules)
        
        pbar = tqdm(total = driver_count, position = 0, leave = True, desc = 'Driver Objects Created')
        #number of drivers per zone
        #use the pickup data to do this
        arrivals_per_zone = pickup_data.sum(axis=1)
        dcounts = np.floor(driver_count * (arrivals_per_zone / arrivals_per_zone.sum()))
        
        driver_index = 0
        for i in dcounts.index:
            for j in range(int(dcounts.loc[i])):
                d = Driver(i, dschedules[driver_index][0], dschedules[driver_index][1])
                #also want to add the driver departure and arrival to the initial event list
                initial_events.append(DriverArrival(d))
                initial_events.append(DriverDeparture(d))
                drivers.append(d)
                pbar.update(1)
                driver_index += 1
        
        for i in range(driver_count - len(drivers)):
            z = np.random.choice(np.arange(1,264))
            d = Driver(z, dschedules[driver_index][0], dschedules[driver_index][1])
            initial_events.append(DriverArrival(d))
            initial_events.append(DriverDeparture(d))
            drivers.append(d)
            pbar.update(1)
            driver_index += 1
                    
        city = City('NYC', np.arange(1,264), drivers, odmatrix)

    event_list = EventList(initial_events)
            
    #iterate through the event list until no events left
    pbar = tqdm(total = arrivals.shape[0], position = 0, leave = True, desc = 'Passengers Processed')
    while not event_list.is_finished():
        
        event = event_list.iterate_next_event()

        result = city.process_event(event)
        if event.type == 'Trip':
            pbar.update(1)

        if result is not None:
            event_list.insert_event(result)    
                
    return passengers, drivers, city, event_list

def simulate_n_days(n,
                    preferred_availability,
                    driver_distribution = 'proportional'):
    #just keep 1 driver history bc it takes up too much memory
    #keep all the waiting time information in dataframes
    passenger_details = []
    driver_history = None
    city_history = None
    
    for i in range(n):
        print(f'--- Day {i} ---')
        arrivals = generate_arrivals_per_zone(show_progress_bar=True)
        p, d, c, e = simulate_with_individual_drivers(arrivals, 
                                                      driver_distribution = driver_distribution, 
                                                      preferred_driver_availability=preferred_availability)
        waiting_times = np.array([(pe.time, pe.start, pe.end, pe.service, pe.waiting_time()) for pe in p])
        waiting_times = pd.DataFrame(waiting_times, columns = ['arrival_time','starting zone', 'ending zone','service_time','waiting_time'])
        waiting_times['arrival_hour'] = waiting_times.arrival_time//60
        waiting_times['replication'] = i
        
        passenger_details.append(waiting_times)
        print(f'Average Waiting Time: {waiting_times.waiting_time.mean()}')
        print(f'Median Waiting Time: {np.median(waiting_times.waiting_time)}')
        print(f'Simulation System Speed: {e.formatted_stats()} \nMore stats: {c.formatted_stats()} \n --- End of Day {i} ---\n')
        
        if i == n - 1:
            driver_history = d
            city_history = c
    
    return pd.concat(passenger_details), driver_history, city_history

if len(sys.argv) == 3:
    print(f'# replications: {sys.argv[1]}')
    print(f'output folder: {sys.argv[2]}')
    num_replications = int(sys.argv[1])
    output_file_name = sys.argv[3]

else:
    num_replications = int(input('Enter number of replications: '))
    output_file_name = input(('Enter output file name: '))

dir_name = 'output/' + output_file_name
if os.path.exists(dir_name):
    os.rmdir(dir_name)
os.mkdir(dir_name)

sys.stdout = Logger(f'{dir_name}/logfile.txt')

minimum_active_trips = load('input_data/minimum_active_uber_trips')
preferred_driver_availability = minimum_active_trips['Driver Count'].values

"""Change the number after num_replications to either preferred_driver_availability or a constant or some other 
   function that records the # of drivers for every minute in the day (0 - 1439)
"""
passenger_details, dhistory, chistory = simulate_n_days(num_replications, 12000)

passenger_details.to_parquet(dir_name + '/passenger_parquet')

unique_driver_dfs = []
i = 0
for d in tqdm(dhistory, position = 0, leave = True, desc = 'Generated Driver Movement Histories'):
    df = d.return_movement_dataframe()
    df['driver_id'] = i
    i += 1
    unique_driver_dfs.append(df)
unique_driver_df = pd.concat(unique_driver_dfs)
unique_driver_df.to_parquet(dir_name + '/driver_histories_parquet')
