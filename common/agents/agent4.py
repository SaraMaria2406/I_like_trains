import random
from common.base_agent import BaseAgent
from common.move import Move
from common.constants import REFERENCE_TICK_RATE

# Student scipers, will be automatically used to evaluate your code
SCIPERS = ["359328", "391236"]


class Agent(BaseAgent):

    def __init__(self, nickname: str, network, logger: str = "client.agent", timeout: float = 1/REFERENCE_TICK_RATE):
        super().__init__(nickname, network, logger)
        self.wall_buffer = 3  # Anticipation buffer for walls, how many cells ahead to check for walls
        self.last_positions = []  # Memory of previous positions to detect if stuck
        self.max_positions_memory = 10
        self.stuck_threshold = 5  # How many repeated positions before considering train is stuck
        self.passenger_group_bonus = 2  # Score multiplier for clustered passengers
        self.prev_trains = {}  # To detect train collisions
        self.forced_direction_change = False  # Force direction change after collision
        self.heading_to_delivery = False  # Track if we're heading to delivery zone
        self.detour_max_distance = 5  # Maximum allowed detour distance in grid cells
        self.detour_passenger_threshold = 2  # Passenger count to consider detour to delivery
        self.delivery_detour_distance = 8  # Max distance to consider detour to delivery
    
    """Methods that implements a memory system to track the train's recent positions and detect if it's stuck in a loop."""
    
    def update_position_history(self):  # 1. Update train position history.
        if self.nickname not in self.all_trains:
            return
        current_pos = tuple(self.all_trains[self.nickname]["position"])
        self.last_positions.append(current_pos)
        if len(self.last_positions) > self.max_positions_memory:
            self.last_positions.pop(0)

    def is_stuck(self):  # 2. Check if train is moving in circles or stuck.
        if len(self.last_positions) < self.stuck_threshold:
            return False
        unique_positions = set(self.last_positions[-self.stuck_threshold:])
        return len(unique_positions) <= 2  # Alternating between 1-2 positions

    """Method that identifies areas where passengers are clustered together."""

    def get_passenger_clusters(self):  # 3. Identify and evaluate passenger clusters on the map.
        if not self.passengers:
            return []
           
        # Convert passenger positions to grid coordinates
        grid_passengers = []
        for p in self.passengers:
            px, py = p["position"]
            grid_passengers.append((px // self.cell_size, py // self.cell_size))
       
        # Evaluate passenger density (how many passengers within radius of 2 cells)
        clusters = []
        for i, (px, py) in enumerate(grid_passengers):
            nearby_count = sum(1 for j, (qx, qy) in enumerate(grid_passengers)
                              if i != j and abs(px - qx) + abs(py - qy) <= 2)
           
            clusters.append({
                "position": (px, py),
                "value": 1 + nearby_count * self.passenger_group_bonus
            })
       
        return sorted(clusters, key=lambda x: x["value"], reverse=True)

    """Methods that handle position tracking and prediction"""

    def get_grid_position(self, train=None):  # 4. Get grid position of a train (default: our train).
        if train is None:
            if self.nickname not in self.all_trains:
                return None
            train = self.all_trains[self.nickname]
           
        pos_x, pos_y = train["position"]
        return (pos_x // self.cell_size, pos_y // self.cell_size)

    def get_occupied_positions(self):  # 5. Get all grid positions occupied by trains and wagons.
        occupied = set()
        for _, train in self.all_trains.items():
            if not train["alive"]:
                continue
               
            train_pos = train["position"]
            occupied.add((train_pos[0] // self.cell_size, train_pos[1] // self.cell_size))
           
            for wagon in train["wagons"]:
                occupied.add((wagon[0] // self.cell_size, wagon[1] // self.cell_size))
               
        return occupied

    def predict_train_positions(self):  # 6. Predict future positions of other trains based on their current directions.
        predicted = set()
       
        for name, train in self.all_trains.items():
            if not train["alive"] or name == self.nickname:
                continue
               
            pos_x, pos_y = train["position"]
            dir_x, dir_y = train["direction"]
           
            # Predict next 5 positions
            for steps in range(1, 6):
                future_x = pos_x + (dir_x * self.cell_size * steps)
                future_y = pos_y + (dir_y * self.cell_size * steps)
               
                grid_x = future_x // self.cell_size
                grid_y = future_y // self.cell_size
               
                predicted.add((grid_x, grid_y))
       
        return predicted

    """Methods that form the core decision-making for individual moves"""

    def is_valid_move(self, grid_x, grid_y, dx, dy, occupied_positions):  # 7. Check if a move is valid (within bounds and not occupied).
        new_x = grid_x + dx
        new_y = grid_y + dy
       
        # Check bounds
        if not (0 <= new_x < self.game_width // self.cell_size and
                0 <= new_y < self.game_height // self.cell_size):
            return False
           
        # Check if position is occupied
        if (new_x, new_y) in occupied_positions:
            return False
           
        return True

    def score_move(self, move, grid_x, grid_y, current_dir, target_pos=None):  # 8. Score a potential move based on multiple factors.
        dx, dy = move.value
        score = 1  # Base score
       
        # CRUCIAL: Avoid U-turns (train cannot make U-turns)
        if (dx, dy) == (-current_dir[0], -current_dir[1]):
            return -float('inf')  # Never allow U-turns
           
        # Bonus for continuing in same direction (momentum)
        if (dx, dy) == current_dir:
            score += 1
           
        # If we have a target, check if this move brings us closer
        if target_pos:
            tx, ty = target_pos
            # Current distance to target
            curr_dist = abs(grid_x - tx) + abs(grid_y - ty)
            # New distance to target after move
            new_dist = abs(grid_x + dx - tx) + abs(grid_y + dy - ty)
           
            if new_dist < curr_dist:
                score += 2  # Bonus for getting closer to target
            elif new_dist > curr_dist:
                score -= 1  # Penalty for moving away from target
       
        return score

    def get_best_move(self, grid_x, grid_y, current_dir, occupied_positions, target_pos=None):  # 9. Get the best move based on scoring all valid moves.
        valid_moves = []
        move_scores = {}
       
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = move.value
           
            # Skip U-turns (train cannot make U-turns)
            if (dx, dy) == (-current_dir[0], -current_dir[1]):
                continue
               
            if self.is_valid_move(grid_x, grid_y, dx, dy, occupied_positions):
                valid_moves.append(move)
                move_scores[move] = self.score_move(move, grid_x, grid_y, current_dir, target_pos)
       
        if valid_moves:
            # Get moves with highest score
            if move_scores:
                max_score = max(move_scores.values())
                best_moves = [m for m, s in move_scores.items() if s == max_score]
                return random.choice(best_moves)
            return random.choice(valid_moves)
           
        return None

    """Methods that implement collision detection"""

    def detect_head_on_collisions(self):  # 10. Detect imminent head-on collisions with other trains.
        if self.nickname not in self.all_trains:
            return False
       
        my_train = self.all_trains[self.nickname]
        if not my_train["alive"]:
            return False
       
        my_pos = my_train["position"]
        my_dir = my_train["direction"]
   
        # Convert to grid coordinates
        my_grid_x = my_pos[0] // self.cell_size
        my_grid_y = my_pos[1] // self.cell_size
   
        # Where I'll be next turn
        next_grid_x = my_grid_x + my_dir[0]
        next_grid_y = my_grid_y + my_dir[1]
   
        # Check each enemy train
        for name, train in self.all_trains.items():
            if name == self.nickname or not train["alive"]:
                continue
           
            enemy_pos = train["position"]
            enemy_dir = train["direction"]
       
            enemy_grid_x = enemy_pos[0] // self.cell_size
            enemy_grid_y = enemy_pos[1] // self.cell_size
       
            enemy_next_x = enemy_grid_x + enemy_dir[0]
            enemy_next_y = enemy_grid_y + enemy_dir[1]
       
            # COLLISION TYPE 1: We're both going to the same cell
            if next_grid_x == enemy_next_x and next_grid_y == enemy_next_y:
                return True
           
            # COLLISION TYPE 2: We're exchanging positions (crossing)
            if next_grid_x == enemy_grid_x and next_grid_y == enemy_grid_y and \
               my_grid_x == enemy_next_x and my_grid_y == enemy_next_y:
                return True
           
            # COLLISION TYPE 3: Head-on approach in same line
            if my_dir[0] != 0 and enemy_dir[0] == -my_dir[0] and my_dir[1] == 0 and enemy_dir[1] == 0:
                # Horizontal head-on
                if my_grid_y == enemy_grid_y:  # Same row
                    distance = abs(my_grid_x - enemy_grid_x)
                    if distance <= 3:  # Critical distance
                        return True
            elif my_dir[1] != 0 and enemy_dir[1] == -my_dir[1] and my_dir[0] == 0 and enemy_dir[0] == 0:
                # Vertical head-on
                if my_grid_x == enemy_grid_x:  # Same column
                    distance = abs(my_grid_y - enemy_grid_y)
                    if distance <= 3:  # Critical distance
                        return True
   
        return False

    def detect_train_collisions(self):  # 11. Detect if a train died at or near our position, indicating a recent collision.
        if not hasattr(self, 'prev_trains') or not self.all_trains or self.nickname not in self.all_trains:
            self.prev_trains = {name: train["alive"] for name, train in self.all_trains.items()}
            return False
           
        my_pos = self.all_trains[self.nickname]["position"]
        my_grid_x = my_pos[0] // self.cell_size
        my_grid_y = my_pos[1] // self.cell_size
       
        for name, train in self.all_trains.items():
            if name != self.nickname and name in self.prev_trains and self.prev_trains[name] and not train["alive"]:
                # This train just died, check if it's close to us
                dead_pos = train["position"]
                dead_grid_x = dead_pos[0] // self.cell_size
                dead_grid_y = dead_pos[1] // self.cell_size
               
                # If dead train is at our position or adjacent, it's likely a collision
                if abs(dead_grid_x - my_grid_x) <= 1 and abs(dead_grid_y - my_grid_y) <= 1:
                    self.logger.debug(f"Train {self.nickname} detected collision with now-dead train {name}!")
                    self.forced_direction_change = True
                    self.prev_trains = {name: train["alive"] for name, train in self.all_trains.items()}
                    return True
       
        # Update train list for next check
        self.prev_trains = {name: train["alive"] for name, train in self.all_trains.items()}
        return False
    
    """Methods that support path planning and opportunistic pickup"""

    def manhattan_distance(self, pos1, pos2):  # 12. Calculate Manhattan distance between two grid positions
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_delivery_zone_center(self):  # 13. Get the center coordinates of the delivery zone
        if not hasattr(self, 'delivery_zone'):
            return None
        
        delivery_pos = self.delivery_zone["position"]
        center_x = (delivery_pos[0] + self.delivery_zone["width"] // 2) // self.cell_size
        center_y = (delivery_pos[1] + self.delivery_zone["height"] // 2) // self.cell_size
        
        return (center_x, center_y)

    def find_passengers_on_path(self, start_pos, end_pos, max_deviation=3):  # 14. Find passengers that are close to the path from start to end position.
        if not self.passengers:
            return None
        
        # Get the direct path distance
        direct_distance = self.manhattan_distance(start_pos, end_pos)
        
        # Find passengers close to the path
        nearby_passengers = []
        for passenger in self.passengers:
            p_pos = (passenger["position"][0] // self.cell_size, 
                     passenger["position"][1] // self.cell_size)
            
            # Calculate distances
            dist_to_passenger = self.manhattan_distance(start_pos, p_pos)
            dist_from_passenger_to_end = self.manhattan_distance(p_pos, end_pos)
            
            # Check if passenger is reasonably close to the path
            # The condition checks if going through the passenger doesn't add too much extra distance
            path_deviation = dist_to_passenger + dist_from_passenger_to_end - direct_distance
            
            if path_deviation <= max_deviation:
                nearby_passengers.append({
                    "position": p_pos,
                    "deviation": path_deviation,
                    "distance": dist_to_passenger  # Distance from current position
                })
        
        if not nearby_passengers:
            return None
        
        # Sort by deviation (prioritize passengers that cause the least detour)
        # Then by distance (closer passengers first)
        nearby_passengers.sort(key=lambda p: (p["deviation"], p["distance"]))
        return nearby_passengers[0]["position"]

    """Methods that from the strategic decision-making core of the agent"""

    def should_detour_to_delivery(self, curr_pos, passenger_count):  # 15. Determine if we should make a detour to delivery zone based on passenger count and distance.
        if not hasattr(self, 'delivery_zone') or passenger_count < self.detour_passenger_threshold:
            return False
            
        delivery_center = self.get_delivery_zone_center()
        if not delivery_center:
            return False
            
        distance_to_delivery = self.manhattan_distance(curr_pos, delivery_center)
        return distance_to_delivery <= self.delivery_detour_distance

    def check_wall_danger(self, steps=1):  # 16. Check if we're heading toward a wall within given steps.
        if self.nickname not in self.all_trains:
            return False
           
        my_train = self.all_trains[self.nickname]
        grid_x, grid_y = self.get_grid_position(my_train)
        dir_x, dir_y = my_train["direction"]
       
        # Check several steps ahead
        for i in range(1, steps + 1):
            next_x = grid_x + (dir_x * i)
            next_y = grid_y + (dir_y * i)
           
            # Wall detection
            if (next_x < 0 or next_x >= self.game_width // self.cell_size or
                next_y < 0 or next_y >= self.game_height // self.cell_size):
                return True
               
        return False

    def find_best_target(self):  # 17. Find the best target position (passenger cluster, detour passenger, or delivery zone).
        if self.nickname not in self.all_trains:
            return None
        
        my_train = self.all_trains[self.nickname]
        passenger_count = len(my_train["wagons"])
        curr_pos = self.get_grid_position()
        
        if curr_pos is None:
            return None
    
        # Initialize delivery zone info
        delivery_center = None
        in_delivery = False
        
        # If we have the delivery zone info
        if hasattr(self, 'delivery_zone'):
            delivery_pos = self.delivery_zone["position"]
            delivery_center = self.get_delivery_zone_center()
            
            # Check if we're already in delivery zone
            delivery_left = delivery_pos[0] // self.cell_size
            delivery_right = (delivery_pos[0] + self.delivery_zone["width"]) // self.cell_size
            delivery_top = delivery_pos[1] // self.cell_size
            delivery_bottom = (delivery_pos[1] + self.delivery_zone["height"]) // self.cell_size
            
            in_delivery = (delivery_left <= curr_pos[0] <= delivery_right and 
                           delivery_top <= curr_pos[1] <= delivery_bottom)
        
        # CASE 1: If we have 4+ passengers, go directly to delivery zone
        if passenger_count >= 4 and delivery_center:
            self.heading_to_delivery = True
            return delivery_center
        
        # CASE 2: If we have any passengers and we're in the delivery zone, stay there until empty
        if passenger_count > 0 and in_delivery and delivery_center:
            # Return a position inside delivery zone but different from current
            # This helps the train "wiggle" around to drop off all passengers
            alt_x = delivery_center[0] + 1 if curr_pos[0] == delivery_center[0] else delivery_center[0]
            alt_y = delivery_center[1] + 1 if curr_pos[1] == delivery_center[1] else delivery_center[1]
            
            # Make sure we stay in bounds of delivery zone
            alt_x = max(delivery_left, min(alt_x, delivery_right))
            alt_y = max(delivery_top, min(alt_y, delivery_bottom))
            
            return (alt_x, alt_y)
        
        # CASE 3: If we're heading to delivery and there's a passenger on the path, go to passenger
        if self.heading_to_delivery and delivery_center and passenger_count < 10:  # Not full
            passenger_on_path = self.find_passengers_on_path(curr_pos, delivery_center, 
                                                           max_deviation=self.detour_max_distance)
            if passenger_on_path:
                self.logger.debug(f"Detouring to pick up passenger at {passenger_on_path} on way to delivery")
                return passenger_on_path
        
        # CASE 4: If we have some passengers and we're close to delivery zone, make a detour
        if self.should_detour_to_delivery(curr_pos, passenger_count) and delivery_center:
            self.logger.debug(f"Detouring to delivery with {passenger_count} passengers")
            self.heading_to_delivery = True
            return delivery_center
    
        # CASE 5: If we don't have any special case, target best passenger cluster
        # Reset heading to delivery flag as we're searching for passengers
        self.heading_to_delivery = False
        
        clusters = self.get_passenger_clusters()
        if not clusters:
            return None
        
        # Find best cluster based on value/distance ratio
        best_score = -float("inf")
        best_target = None
    
        for cluster in clusters:
            px, py = cluster["position"]
            dist = self.manhattan_distance(curr_pos, (px, py))
            
            if dist == 0:
                continue
            
            score = cluster["value"] / dist
            
            if score > best_score:
                best_score = score
                best_target = (px, py)
            
        return best_target

    """Get_move method bring everything together in a well-structured decision hierarchy"""

    def get_move(self):  # 18. Main decision method to get the next move for the train.
        # Update position history
        self.update_position_history()
   
        # Basic checks
        if self.nickname not in self.all_trains or not self.all_trains[self.nickname]["alive"]:
            return random.choice(list(Move))

        # Get train info
        my_train = self.all_trains[self.nickname]
        grid_x, grid_y = self.get_grid_position(my_train)
        current_dir = my_train["direction"]
       
        # Get all occupied and predicted positions
        occupied_positions = self.get_occupied_positions()
        occupied_positions.update(self.predict_train_positions())
       
        # Find best target position (this now includes detour logic)
        target_pos = self.find_best_target()
       
        # Critical checks in priority order
        critical_conditions = [
            # 1. Head-on collision detection (highest priority)
            self.detect_head_on_collisions(),
           
            # 2. Stuck detection or forced direction change after collision
            self.is_stuck() or self.forced_direction_change or self.detect_train_collisions(),
           
            # 3. Wall danger (within 3 cells)
            self.check_wall_danger(steps=self.wall_buffer),
           
            # 4. Train collision danger (immediate next cell)
            (grid_x + current_dir[0], grid_y + current_dir[1]) in occupied_positions
        ]
       
        if any(critical_conditions):
            # Reset forced direction flag if it was set
            self.forced_direction_change = False
           
            # Get best move in critical situation
            critical_move = self.get_best_move(grid_x, grid_y, current_dir, occupied_positions, target_pos)
            if critical_move:
                self.logger.debug(f"Train {self.nickname} making critical move: {critical_move}")
                return critical_move
       
        # If we have a target and no critical situation, move toward target
        if target_pos:
            tx, ty = target_pos
            dx = tx - grid_x
            dy = ty - grid_y
           
            # Try horizontal move first if needed
            if dx != 0:
                move = Move.RIGHT if dx > 0 else Move.LEFT
                if self.is_valid_move(grid_x, grid_y, move.value[0], move.value[1], occupied_positions):
                    return move
                   
            # Try vertical move if needed
            if dy != 0:
                move = Move.DOWN if dy > 0 else Move.UP
                if self.is_valid_move(grid_x, grid_y, move.value[0], move.value[1], occupied_positions):
                    return move
       
        # Continue in current direction if it's safe
        next_x = grid_x + current_dir[0]
        next_y = grid_y + current_dir[1]
        if (0 <= next_x < self.game_width // self.cell_size and
            0 <= next_y < self.game_height // self.cell_size and
            (next_x, next_y) not in occupied_positions):
           
            # Find current move enum
            for move in Move:
                if move.value == current_dir:
                    return move
       
        # Fallback: find any safe move
        fallback_move = self.get_best_move(grid_x, grid_y, current_dir, occupied_positions)
        if fallback_move:
            return fallback_move
           
        # Last resort: try any move except U-turn
        for move in Move:
            if move.value != (-current_dir[0], -current_dir[1]):  # Avoid U-turn
                return move
               
        # Ultimate fallback
        return random.choice([move for move in Move if move.value != (-current_dir[0], -current_dir[1])])