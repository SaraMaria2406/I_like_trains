import random
from common.base_agent import BaseAgent
from common.move import Move

# Student scipers, will be automatically used to evaluate your code
SCIPERS = ["359328", "391236"]


class Agent(BaseAgent):

    def __init__(self, nickname: str, network, logger: str = "client.agent"):
        super().__init__(nickname, network, logger)
        self.wall_buffer = 2

    def move_to_closest_passenger(self):
        if not self.passengers or self.nickname not in self.all_trains:
            return None
        # converting my positions to grid coordinates by dividing by cell_size
        my_pos = self.all_trains[self.nickname]["position"]
        my_x = my_pos[0] // self.cell_size
        my_y = my_pos[1] // self.cell_size

        # Find the closest passenger
        min_distance = float("inf")
        closest_passenger = None

        for p in self.passengers:
            px, py = p["position"]
            px //= self.cell_size
            py //= self.cell_size
            dist = abs(px - my_x) + abs(py - my_y)  # Manhattan distance
            if dist < min_distance:
                min_distance = dist
                closest_passenger = (px, py)

        if closest_passenger is None:
            return None

        px, py = closest_passenger
        dx = px - my_x
        dy = py - my_y

        # moving horizontally and then vertically
        if dx != 0:
            move = Move.RIGHT if dx > 0 else Move.LEFT
        elif dy != 0:
            move = Move.DOWN if dy > 0 else Move.UP
        else:
            return None  # Already at the passenger
            
        self.logger.debug(f"Train {self.nickname} heading toward passenger at {closest_passenger}")

        # Validate moves to make sure we stay in bounds
        test_x = my_x + move.value[0]
        test_y = my_y + move.value[1]
        if (0 <= test_x < self.game_width // self.cell_size and 
            0 <= test_y < self.game_height // self.cell_size):
            return move

        return None
    
    
    def go_to_delivery_zone(self):

        
        """Navigate to the delivery zone when carrying 5 or more passengers.
        Uses self.delivery_zone with proper attributes (position, height, width).
        Returns a Move if heading to delivery zone, None otherwise."""
        
        # Check if we exist and if delivery_zone exists
        if self.nickname not in self.all_trains or not hasattr(self, 'delivery_zone'):
            return None
    
        # Get this train's position
        my_train_data = self.all_trains[self.nickname]
        my_train_position = my_train_data["position"]
    
        # Check how many passengers this train is currently carrying
        onboard_count = len(self.all_trains[self.nickname]["wagons"])
        if onboard_count < 3:
            return None


        self.logger.debug(f"Train {self.nickname} has {onboard_count} passengers and is heading to the delivery zone.")
    
        # Get train position in grid coordinates
        my_x = my_train_position[0] // self.cell_size
        my_y = my_train_position[1] // self.cell_size
    
        # Use self.delivery_zone to get the position, height and width
        delivery_pos = self.delivery_zone["position"]
        delivery_height = self.delivery_zone["height"]
        delivery_width = self.delivery_zone["width"]
    
        # Calculate the center of the delivery zone in grid coordinates
        delivery_center_x = (delivery_pos[0] + delivery_width // 2) // self.cell_size
        delivery_center_y = (delivery_pos[1] + delivery_height // 2) // self.cell_size
    
        # Calculate direction to move to delivery zone
        dx = delivery_center_x - my_x
        dy = delivery_center_y - my_y
    
        self.logger.debug(f"Train {self.nickname} at ({my_x},{my_y}), delivery target: ({delivery_center_x},{delivery_center_y})")

        # Prioritize horizontal movement first, then vertical
        if dx != 0:
            move = Move.RIGHT if dx > 0 else Move.LEFT
        elif dy != 0:
            move = Move.DOWN if dy > 0 else Move.UP
        else:
            # We've arrived at the delivery zone
            self.logger.debug(f"Train {self.nickname} arrived at delivery zone")
            return None
    
        # Validate move is within bounds
        test_x = my_x + move.value[0]
        test_y = my_y + move.value[1]
        if (0 <= test_x < self.game_width // self.cell_size and
            0 <= test_y < self.game_height // self.cell_size):
            return move
    
        return None

    def avoiding_trains(self):  
        """
        Returns a direction to avoid crashing into other trains.
        If no collision is predicted, returns None.
        """
        my_train = self.all_trains[self.nickname]
        current_x, current_y = my_train["position"]
        dir_x, dir_y = my_train["direction"]

        # Rassemble toutes les positions occupées par les autres trains
        occupied_positions = set()

        for name, train in self.all_trains.items():
            if not train["alive"]:
                continue  # Ignore les trains morts

            occupied_positions.add(tuple(train["position"]))  # tête du train

            for wagon in train["wagons"]:
                occupied_positions.add(tuple(wagon))

        # Calcul de la prochaine position si on continue tout droit
        next_x = current_x + dir_x * self.cell_size
        next_y = current_y + dir_y * self.cell_size
        next_pos = (next_x, next_y)

        # Si on fonce dans un train, change de direction
        if next_pos in occupied_positions:
            safe_directions = []

            for move_option in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dx, dy = move_option.value
                new_x = current_x + dx * self.cell_size
                new_y = current_y + dy * self.cell_size

                if ((dx, dy) == (dir_x, dir_y) or (dx, dy) == (-dir_x, -dir_y)):
                    continue  # Ignore direction actuelle ou demi-tour

                if (0 <= new_x < self.game_width and 0 <= new_y < self.game_height):
                    if (new_x, new_y) not in occupied_positions:
                        safe_directions.append(move_option)

            if safe_directions:
                return random.choice(safe_directions)

        return None


    def avoiding_walls(self):
        """
        Checks if the train is about to hit a wall and returns a safe direction.
        Uses a safety buffer to detect walls earlier.
        """
        if not self.all_trains or self.nickname not in self.all_trains:
            return None
        
        # Get current position and direction
        my_train = self.all_trains[self.nickname]
        current_x, current_y = my_train["position"]
        dir_x, dir_y = my_train["direction"]
    
        # Check several steps ahead for wall detection (using wall_buffer)
        for steps in range(1, self.wall_buffer + 1):
            next_x = current_x + (dir_x * self.cell_size * steps)
            next_y = current_y + (dir_y * self.cell_size * steps)
            
            # If we're approaching a wall, we need to change direction
            if (next_x < 0 or next_x >= self.game_width or
                next_y < 0 or next_y >= self.game_height):
                
                self.logger.debug(f"Train {self.nickname} avoiding wall at {current_x},{current_y}")
                
                # Calculate available safe directions
                available_directions = []
                
                # Check each potential direction to see if it's safe
                for move_option in [Move.RIGHT, Move.LEFT, Move.UP, Move.DOWN]:
                    # Don't consider current direction or its opposite
                    if move_option.value == (dir_x, dir_y) or move_option.value == (-dir_x,-dir_y):
                        continue

                    """if ((move.value[0] == dir_x and move.value[1] == dir_y) or
                        (move.value[0] == -dir_x and move.value[1] == -dir_y)):
                        continue"""
                    
                    # Check if this direction leads to a wall
                    test_x = current_x + move_option.value[0] * self.cell_size
                    test_y = current_y + move_option.value[1] * self.cell_size
                    
                    if (0 <= test_x < self.game_width and 0 <= test_y < self.game_height):
                        available_directions.append(move_option)
                        self.logger.debug(f"Available directions: {available_directions}")

                # Choose a random safe direction if available
                if available_directions:
                    new_direction = random.choice(available_directions)
                    
                    # Checking if the train has no valid direction
                    self.logger.debug(f"Train {self.nickname} changing direction to {new_direction}")

                    return new_direction
                
                # Checking if the train has no valid direction

        
        # No wall detected within our buffer
        return None

    def get_move(self):
        """
        Called regularly to get the next move for your train.
        Balanced approach between safety and efficiency.
        """
        if self.nickname not in self.all_trains:
            return random.choice(list(Move))
    
        # Check if we're about to hit a wall in the immediate future
        # This is critical safety check that overrides other behaviors
        my_train = self.all_trains[self.nickname]
        current_pos = my_train["position"]
        current_dir = my_train["direction"]
    
        # Check only immediate danger (1 step ahead)
        next_x = current_pos[0] + (current_dir[0] * self.cell_size)
        next_y = current_pos[1] + (current_dir[1] * self.cell_size)
        immediate_danger = (next_x < 0 or next_x >= self.game_width or
                            next_y < 0 or next_y >= self.game_height)
    
        if immediate_danger:
            # Emergency wall avoidance - immediate danger
            new_direction = self.avoiding_walls()
            if new_direction:
                self.logger.debug(f"Train {self.nickname} in DANGER - emergency wall avoidance {new_direction}")
                return new_direction
    
        # Second priority: avoid other trains if in immediate danger
        new_direction = self.avoiding_trains()
        if new_direction:
            self.logger.debug(f"Train {self.nickname} is avoiding train with move {new_direction}")
            return new_direction
    
        # Third priority: deliver passengers when we have enough
        to_delivery = self.go_to_delivery_zone()
        if to_delivery:
            # Safety check to avoid running into walls while going to delivery zone
            test_x = current_pos[0] + (to_delivery.value[0] * self.cell_size)
            test_y = current_pos[1] + (to_delivery.value[1] * self.cell_size)
        
            if (0 <= test_x < self.game_width and 0 <= test_y < self.game_height):
                self.logger.debug(f"Train {self.nickname} is going to delivery zone with move {to_delivery}")
                return to_delivery
    
        # Fourth priority: pick up passengers when not delivering
        to_passenger = self.move_to_closest_passenger()
        if to_passenger:
            # Safety check to avoid running into walls while going to passenger
            test_x = current_pos[0] + (to_passenger.value[0] * self.cell_size)
            test_y = current_pos[1] + (to_passenger.value[1] * self.cell_size)
        
            if (0 <= test_x < self.game_width and 0 <= test_y < self.game_height):
                self.logger.debug(f"Train {self.nickname} heading to passenger with move {to_passenger}")
                return to_passenger
    
        # Check for wall avoidance with longer look-ahead
        # This is proactive wall avoidance but with lower priority
        if not immediate_danger:
            new_direction = self.avoiding_walls()
            if new_direction:
                self.logger.debug(f"Train {self.nickname} proactively avoiding wall with move {new_direction}")
                return new_direction
    
        # If no special action needed, maintain current direction
        for m in Move:
            if m.value == current_dir:
                return m
    


        """
        Called regularly called to get the next move for your train. Implement
        an algorithm to control your train here. You will be handing in this file.

        For now, the code simply picks a random direction between UP, DOWN, LEFT, RIGHT

        This method must return one of moves.MOVE
        """

