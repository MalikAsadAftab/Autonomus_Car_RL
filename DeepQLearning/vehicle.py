import numpy as np

class Vehicle:
    def __init__(self, size, is_dynamic=True):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)
        self.is_dynamic = is_dynamic  # True if the obstacle can move, False if static
        self.directions = {
            'N': (-1, 0), 'S': (1, 0),
            'E': (0, 1), 'W': (0, -1),
            'NE': (-1, 1), 'NW': (-1, -1),
            'SE': (1, 1), 'SW': (1, -1)
        }
    
    def __sub__(self, other):
        # Returns the relative position to another vehicle
        return (self.x - other.x, self.y - other.y)

    def move(self, direction):
        # Update the vehicle position based on the direction
        if direction in self.directions:
            dx, dy = self.directions[direction]
            self.x = max(0, min(self.x + dx, self.size - 1))
            self.y = max(0, min(self.y + dy, self.size - 1))

    def random_move(self):
        # Moves the vehicle in a random direction
        direction = np.random.choice(list(self.directions.keys()))
        self.move(direction)

    def action(self, choice):
        if not self.is_dynamic:
            return  # No action if the obstacle is static
        # Movement logic for dynamic obstacles
        movements = [(1, 1), (-1, -1), (-1, 1), (1, -1)]
        dx, dy = movements[choice]
        # Movement logic with smaller steps for each direction
        if choice == 0:  # Move right
            self.x = min(self.size - 1, self.x + 1)
        elif choice == 1:  # Move left
            self.x = max(0, self.x - 1)
        elif choice == 2:  # Move up
            self.y = max(0, self.y - 1)
        elif choice == 3:  # Move down
            self.y = min(self.size - 1, self.y + 1)
    
def initialize_obstacles(num_dynamic, num_static, size):
    obstacles = [Vehicle(size, is_dynamic=True) for _ in range(num_dynamic)]
    obstacles += [Vehicle(size, is_dynamic=False) for _ in range(num_static)]
    return obstacles
