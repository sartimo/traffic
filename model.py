import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from safetensors.torch import save_file
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Define constants
sourceLocations = ["south", "east", "west", "north"]
directions = ["straight", "right"]
vehiclerate = 100
NUM_EPOCHS = 5000

# Traffic light states and durations
traffic_light_states = ["green", "yellow", "red"]
traffic_light_duration = [0.05, 0.02, 0.05]  # Duration in seconds for each state

# Counters
traffic_density_counter = 0
red_light_counter = 0
waiting_vehicles = {"north": 0, "south": 0, "east": 0, "west": 0}  # Count for each direction
red_light_queue = {"north": [], "south": [], "east": [], "west": []}  # Queue for each direction

# Define the FFNN Model
class TrafficLightController(nn.Module):
    def __init__(self):
        super(TrafficLightController, self).__init__()
        self.fc1 = nn.Linear(5, 64)  # 4 traffic densities + 1 red light counter
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)  # Output: Action for each traffic light (0: red, 1: green)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation function at output layer for action values
        return x

# Generate dummy data for training (Replace with your real data)
def generate_dummy_data(num_samples):
    data = []
    for _ in range(num_samples):
        traffic_density = np.random.randint(0, 10, size=4)  # 4 directions
        red_light_counter = np.random.randint(0, 20)  # Random red light counter
        # Random actions (0: red, 1: green) for each traffic light
        action = np.random.randint(0, 2, size=4)  
        data.append((np.concatenate([traffic_density, [red_light_counter]]), action))
    return data

# Prepare the dataset
dataset = generate_dummy_data(1000)

# Normalize inputs
scaler = StandardScaler()
inputs = scaler.fit_transform(np.array([sample[0] for sample in dataset]))
targets = np.array([sample[1] for sample in dataset])

# Create tensors
inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)

# Create DataLoader
train_data = torch.utils.data.TensorDataset(inputs, targets)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)  # Increased batch size

# Initialize the model, loss function, and optimizer
model = TrafficLightController()
criterion = nn.MSELoss()  # Mean Squared Error for regression output
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)  # Adjusted learning rate and added weight decay

# Initialize lists for logging
losses = []
accuracies = []

# Variables to track the best epoch
best_accuracy = 0
best_loss = float('inf')
best_epoch_accuracy = 0
best_epoch_loss = 0

# Training Loop with Gradient Clipping
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    correct_predictions = 0
    
    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        predicted = torch.round(torch.sigmoid(outputs))  # Rounding to get binary actions
        correct_predictions += (predicted == batch_targets).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    
    accuracy = correct_predictions / (len(batch_inputs) * len(train_loader))  # Calculate accuracy
    accuracies.append(accuracy)

    # Check for best accuracy and loss
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_epoch_accuracy = epoch

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_epoch_loss = epoch

    if epoch % 50 == 0:
        print(f'Training Epoch [{epoch}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

# Save the model in safetensors format
save_file(model.state_dict(), f"output/traffic-{NUM_EPOCHS}.safetensors")

# Save training log to CSV
training_log = pd.DataFrame({'Epoch': range(NUM_EPOCHS), 'Loss': losses, 'Accuracy': accuracies})
training_log.to_csv('training.csv', index=False)

# Plotting the training loss and accuracy
plt.figure(figsize=(14, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(losses, label='Training Loss', color='blue')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Training Accuracy', color='orange')
plt.title('Training Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training.png')

# Prepare the data for Safetensors
best_info = {
    "best_epoch_accuracy": torch.tensor(best_epoch_accuracy),
    "best_accuracy": torch.tensor(best_accuracy),
    "best_epoch_loss": torch.tensor(best_epoch_loss),
    "best_loss": torch.tensor(best_loss)
}

# Save best accuracy and loss information in safetensors
save_file(best_info, f'output/traffic-{NUM_EPOCHS}-best-epoch.safetensors')

# After the simulation
print(f"Model saved in 'output/traffic-{NUM_EPOCHS}.safetensors'.")
print(f"Best Accuracy: {best_accuracy:.4f} at Epoch: {best_epoch_accuracy}")
print(f"Lowest Training Loss: {best_loss:.4f} at Epoch: {best_epoch_loss}")

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

# Function to control traffic lights based on the model
def control_traffic_lights(traffic_density, red_light_counter):
    input_tensor = torch.tensor(np.concatenate([traffic_density, [red_light_counter]]), dtype=torch.float32).unsqueeze(0)
    action_values = model(input_tensor).detach().numpy()[0]
    
    # Choose actions based on the model's output
    actions = np.argmax(action_values)  # Get the index of the highest value
    return actions  # Return the index of the direction to turn green

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

    # Gather traffic density and red light counter
    current_density = [waiting_vehicles["north"], waiting_vehicles["south"],
                       waiting_vehicles["east"], waiting_vehicles["west"]]
    current_light = next(light_cycles[src])

    # Use the model to control the traffic lights based on current conditions
    action = control_traffic_lights(current_density, red_light_counter)

    # Check if the light is green for the chosen direction based on the model's action
    if action == sourceLocations.index(src):  # If the model says this light should be green
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
