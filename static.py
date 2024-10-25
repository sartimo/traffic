import random
import time

# Define constants
sourceLocations = ["south", "east", "west", "north"]
directions = ["straight", "right"]
vehiclerate = 100

# Traffic light states and durations
traffic_light_states = ["green", "yellow", "red"]
traffic_light_duration = [0.05, 0.02, 0.05]  # Duration in seconds for each state

# Counters
traffic_density_counter = 0
red_light_counter = 0
waiting_vehicles = {"north": 0, "south": 0, "east": 0, "west": 0}  # Count for each direction
red_light_queue = {"north": [], "south": [], "east": [], "west": []}  # Queue for each direction

def map_direction(value, direction):
    straight_map = {
        'north': 'south',
        'south': 'north',
        'east': 'west',
        'west': 'east'
    }

    right_map = {
        'south': 'east',
        'east': 'north',
        'north': 'west',
        'west': 'south'
    }

    if direction == "straight":
        return straight_map.get(value.lower(), "Invalid direction")
    elif direction == "right":
        return right_map.get(value.lower(), "Invalid direction")
    else:
        return "Invalid input"

def traffic_light_cycle():
    """Simulates the traffic light cycle for each direction."""
    while True:
        for state, duration in zip(traffic_light_states, traffic_light_duration):
            yield state
            time.sleep(duration)  # Wait for the duration of the current light state

# Create traffic light generators for each direction
light_cycles = {
    "north": traffic_light_cycle(),
    "south": traffic_light_cycle(),
    "east": traffic_light_cycle(),
    "west": traffic_light_cycle()
}

# Run the simulation
for i in range(vehiclerate + 1):
    vehicleid = i

    # Choose a random source and direction
    src = random.choice(sourceLocations)
    direction = random.choice(directions)

    # Determine the route and final destination based on the source and the chosen direction
    route = map_direction(src, direction)
    dest = route  # Destination now aligns with the chosen route

    # Randomly pick a vehicle type
    vehicletypes = ["car", "truck", "bike", "cyclist"]
    vehicle = random.choice(vehicletypes)

    # Get the current traffic light state for the vehicle's direction
    current_light = next(light_cycles[src])

    # Check if the light is green and release any waiting vehicles
    if current_light == "green":
        # First, check if there are vehicles waiting at the red light
        if red_light_queue[src]:
            for waiting_vehicle in red_light_queue[src]:
                vid, vtype, vsrc, vdest = waiting_vehicle
                print(f"{vid}: {vtype} was waiting at the {src} traffic light from {vsrc} to {vdest}. Now the light is green, so it proceeds to {vdest}.")
            # Clear the queue after all vehicles proceed
            red_light_queue[src].clear()
            waiting_vehicles[src] = 0  # Reset waiting vehicles counter

        # Now process the current vehicle since the light is green
        print(f"{vehicleid}: {vehicle} spawned from {src} and goes to {route}. The {src} traffic light is green.")
        print(f"{vehicleid}: {vehicle} has arrived at the destination: {dest}.")
    elif current_light == "red":
        # Add vehicle to red light queue for its direction
        red_light_queue[src].append((vehicleid, vehicle, src, dest))

        # Increment counters
        waiting_vehicles[src] += 1
        red_light_counter += 1

        # Increment traffic density counter if more than 3 vehicles are waiting in any direction
        if waiting_vehicles[src] > 3:
            traffic_density_counter += 1

        print(f"{vehicleid}: {vehicle} spawned from {src} and goes to {route}. The {src} traffic light is red, so the vehicle has to stop.")
    else:
        # If the light is yellow, the vehicle can proceed but may need to stop soon
        print(f"{vehicleid}: {vehicle} spawned from {src} and goes to {route}. The {src} traffic light is {current_light}.")
        print(f"{vehicleid}: {vehicle} has arrived at the destination: {dest}.")

# After the simulation
print(f"Total Traffic Density Incidents: {traffic_density_counter}")
print(f"Total Vehicles Stopped at Red Light: {red_light_counter}")

