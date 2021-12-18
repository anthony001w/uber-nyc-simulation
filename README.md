# uber-nyc-simulation

## Simulating
Need to create an output folder in the same directory as run_replications.py. To run simulation replications, just type 'python3 run_replications.py' and specify the # of replications and the directory. To change the simulation parameters, you'll need to go into the script and make changes where specified. The most important change is the driver availability function (an input to the function simulate_n_days)

## Animation
'python3 nycuberviz.py {FRAMES PER SECOND} {SPEED OF SIMULATION} {DIRECTORY IN OUTPUT/} {random/lines}'

ex) 'python3 nycuberviz.py 60 15 directory_name random'

{FRAMES PER SECOND} is self explanatory Ex) 60
{SPEED OF SIMULATIONS} is how fast the drivers move in the simulation, basically speed of simulations * 60 is how many seconds pass in system vs. seconds in real time.
{DIRECTORY} is just which run you want to use
{random/lines} specifies where drivers go to and from on screen, random places drivers randomly in their zones while lines means drivers going to a zone only go to one point in that zone
