import random
from common.base_agent import BaseAgent
from common.move import Move
from common.constants import REFERENCE_TICK_RATE


# Student scipers, will be automatically used to evaluate your code
SCIPERS = ["359328", "391236"]


class Agent(BaseAgent):

    def __init__(self, nickname: str, network, logger: str = "client.agent", timeout: float = 1/REFERENCE_TICK_RATE):
        super().__init__(nickname, network, logger, timeout)
        self.wall_buffer = 3  # Zone tampon pour anticiper les murs
        self.last_positions = []  # Pour détecter les boucles de mouvement
        self.max_positions_memory = 10  # Nombre maximum de positions à mémoriser
        self.stuck_threshold = 5  # Nombre de positions avant de considérer le train comme bloqué
        self.passenger_group_bonus = 2  # Bonus pour les groupes de passagers
        self.prev_trains = {}  # Pour détecter les collisions avec d'autres trains
        self.forced_direction_change = False  # Force un changement de direction après collision

    #---------- Methods that implements a memory system to track the train's recent posisiton ----------

    def update_position_history(self):
        """
        Update train position history.
        """
        if self.nickname not in self.all_trains:
            return
        current_pos = tuple(self.all_trains[self.nickname]["position"])
        self.last_positions.append(current_pos)
        if len(self.last_positions) > self.max_positions_memory:
            self.last_positions.pop(0)  # Supprime la position la plus ancienne

    def is_stuck(self):
        """
        Check if train is moving in circles or stuck.
        """
        if len(self.last_positions) < self.stuck_threshold:
            return False
        unique_positions = set(self.last_positions[-self.stuck_threshold:])
        return len(unique_positions) <= 2  # Considère bloqué si alterne entre 1-2 positions

    # --------------------------------------------------------------------------------------------------

    # -------------- Method that identifies areas where passengers are clustered together --------------

    def get_passenger_clusters(self):
        """
        Identify and evaluate passenger clusters on the map.
        """
        if not self.passengers:
            return []
           
        # Convertit les positions des passagers en coordonnées de grille
        grid_passengers = []
        for p in self.passengers:
            px, py = p["position"]
            grid_passengers.append((px // self.cell_size, py // self.cell_size))
       
        # Évalue la densité de passagers (nombre de passagers dans un rayon de 2 cellules)
        clusters = []
        for i, (px, py) in enumerate(grid_passengers):
            nearby_count = sum(1 for j, (qx, qy) in enumerate(grid_passengers)
                              if i != j and abs(px - qx) + abs(py - qy) <= 2)
           
            clusters.append({
                "position": (px, py),
                "value": 1 + nearby_count * self.passenger_group_bonus  # Valorise les groupes denses
            })
       
        return sorted(clusters, key=lambda x: x["value"], reverse=True)  # Trie par valeur décroissante

    # --------------------------------------------------------------------------------------------------

    # ------------------------------ Position extraction methods ---------------------------------------

    def get_grid_position(self, train=None):
        """
        Get grid position of a train (default: our train).
        """
        if train is None:
            if self.nickname not in self.all_trains:
                return None
            train = self.all_trains[self.nickname]
           
        pos_x, pos_y = train["position"]
        return (pos_x // self.cell_size, pos_y // self.cell_size)  # Convertit en coordonnées de grille

    def get_occupied_positions(self):
        """
        Get all grid positions occupied by trains and wagons.
        """
        occupied = set()
        for _, train in self.all_trains.items():
            if not train["alive"]:
                continue
               
            train_pos = train["position"]
            occupied.add((train_pos[0] // self.cell_size, train_pos[1] // self.cell_size))
           
            for wagon in train["wagons"]:
                occupied.add((wagon[0] // self.cell_size, wagon[1] // self.cell_size))
               
        return occupied

    def predict_train_positions(self):
        """
        Predict future positions of other trains based on their current directions.
        """
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

    # ------------------------------------------------------------------------------------------------- 

    # ---------- Methods that form the core decision-making for individual moves ----------------------

    def is_valid_move(self, grid_x, grid_y, dx, dy, occupied_positions):
        """
        Check if a move is valid (within bounds and not occupied).
        """
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

    def score_move(self, move, grid_x, grid_y, current_dir, target_pos=None):
        """
        Score a potential move based on multiple factors.
        """
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

    def get_best_move(self, grid_x, grid_y, current_dir, occupied_positions, target_pos=None):
        """
        Get the best move based on scoring all valid moves.
        """
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
                return random.choice(best_moves)  # Choose a random move in the better ones
            return random.choice(valid_moves)
           
        return None  
    
    # --------------------------------------------------------------------------------------------------

    # ------------------------ Methods that implement collision detection ------------------------------

    def detect_head_on_collisions(self):
        """
        Detect imminent head-on collisions with other trains.
        """
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
                if my_grid_x == enemy_grid_x:  # same column
                    distance = abs(my_grid_y - enemy_grid_y)
                    if distance <= 3:  # Critical distance
                        return True
   
        return False

    def detect_train_collisions(self):
        """
        Detect if a train died at or near our position, indicating a recent collision.
        """
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
                    self.logger.debug(f"Train {self.nickname} a détecté une collision avec le train maintenant mort {name}!")
                    self.forced_direction_change = True
                    self.prev_trains = {name: train["alive"] for name, train in self.all_trains.items()}
                    return True
       
        # Update train list for next check
        self.prev_trains = {name: train["alive"] for name, train in self.all_trains.items()}
        return False
    
    # -------------------------------------------------------------------------------------------------------

    # -------------------------------- Passenger management and targeting -----------------------------------

    def check_adjacent_passenger_cluster(self, grid_x, grid_y):
        """
        Checks if there's a group of passengers adjacent or very close to our position
        """
        # Gets all passenger groups
        clusters = self.get_passenger_clusters()
   
        # Checks for passengers adjacent or within 1-2 cells radius
        opportunistic_range = 2
   
        for cluster in clusters:
            px, py = cluster["position"]
            dist = abs(px - grid_x) + abs(py - grid_y)
       
            if 0 < dist <= opportunistic_range:
                return (px, py)
           
        return None

    def find_best_target(self):
        """
        Finds the best target (passenger group or delivery zone) with opportunistic behavior
        """
        if self.nickname not in self.all_trains:
            return None
   
        my_train = self.all_trains[self.nickname]
        passenger_count = len(my_train["wagons"])

        # Gets current position
        grid_x, grid_y = self.get_grid_position()
        if grid_x is None:
            return None

        # Identifies passenger groups
        clusters = self.get_passenger_clusters()
   
        # If delivery zone info is available
        if hasattr(self, 'delivery_zone'):
            delivery_pos = self.delivery_zone["position"]
            delivery_center_x = (delivery_pos[0] + self.delivery_zone["width"] // 2) // self.cell_size
            delivery_center_y = (delivery_pos[1] + self.delivery_zone["height"] // 2) // self.cell_size
   
            # Checks if already in delivery zone
            delivery_left = delivery_pos[0] // self.cell_size
            delivery_right = (delivery_pos[0] + self.delivery_zone["width"]) // self.cell_size
            delivery_top = delivery_pos[1] // self.cell_size
            delivery_bottom = (delivery_pos[1] + self.delivery_zone["height"]) // self.cell_size
   
            in_delivery = (delivery_left <= grid_x <= delivery_right and
                        delivery_top <= grid_y <= delivery_bottom)
   
            # Checks if delivery zone is close (less than 3 cells away)
            delivery_nearby = (abs(grid_x - delivery_center_x) + abs(grid_y - delivery_center_y)) <= 3
       
            # If has passengers and is in or near delivery zone, prioritize delivery
            if passenger_count > 0 and (in_delivery or delivery_nearby):
                self.logger.debug(f"Train {self.nickname} est près de la zone de livraison avec {passenger_count} passagers, priorité à la livraison")
                # Returns a position in delivery zone but different from current if already inside
                if in_delivery:
                    alt_x = delivery_center_x + 1 if grid_x == delivery_center_x else delivery_center_x
                    alt_y = delivery_center_y + 1 if grid_y == delivery_center_y else delivery_center_y
               
                    # Ensures staying within delivery zone boundaries
                    alt_x = max(delivery_left, min(alt_x, delivery_right))
                    alt_y = max(delivery_top, min(alt_y, delivery_bottom))
               
                    return (alt_x, alt_y)
                else:
                    return (delivery_center_x, delivery_center_y)
       
            # If has 4+ passengers, head to delivery zone unless a passenger is right next to it
            if passenger_count >= 4:
                # Checks if there's an adjacent passenger group
                adjacent_cluster = self.check_adjacent_passenger_cluster(grid_x, grid_y)
                if adjacent_cluster and passenger_count < 10:  # Limite pour éviter surcharge
                    self.logger.debug(f"Train {self.nickname} a trouvé un passager adjacent malgré {passenger_count} passagers, comportement opportuniste")
                    return adjacent_cluster
                return (delivery_center_x, delivery_center_y)

        # Checks adjacent passenger groups (opportunistic pickup)
        adjacent_cluster = self.check_adjacent_passenger_cluster(grid_x, grid_y)
        if adjacent_cluster:
            self.logger.debug(f"Train {self.nickname} ramasse un passager adjacent de façon opportuniste")
            return adjacent_cluster
       
        # If empty or impossible to deliver, target the best passenger group
        if not clusters:
            return None
   
        # Finds the best group based on value/distance ratio
        best_score = -float("inf")
        best_target = None

        for cluster in clusters:
            px, py = cluster["position"]
            dist = abs(px - grid_x) + abs(py - grid_y)
   
            if dist == 0:
                continue
       
            score = cluster["value"] / dist
   
            if score > best_score:
                best_score = score
                best_target = (px, py)
       
        return best_target
    
    # ---------------------------------------------------------------------------------------------------------
    
    # ------------------------------ Collision and safety management ------------------------------------------
   
    def check_wall_danger(self, steps=1):
        """
        Checks if we're heading toward a wall in the next steps
        """
        if self.nickname not in self.all_trains:
            return False
           
        my_train = self.all_trains[self.nickname]
        grid_x, grid_y = self.get_grid_position(my_train)
        dir_x, dir_y = my_train["direction"]
       
        # Checks several steps ahead
        for i in range(1, steps + 1):
            next_x = grid_x + (dir_x * i)
            next_y = grid_y + (dir_y * i)
           
            # Wall detection
            if (next_x < 0 or next_x >= self.game_width // self.cell_size or
                next_y < 0 or next_y >= self.game_height // self.cell_size):
                return True
               
        return False

    # ---------------------------------------------------------------------------------------------------------

    # ----------- Get_move method bring everything together in a well-structured decision hierarchy -----------

    def get_move(self):
        """
        Main decision method to get the next move for the train.
        """
        self.update_position_history()  # Update position history
   
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
            # a. Head-on collision detection (highest priority)
            self.detect_head_on_collisions(),
           
            # b. Stuck detection or forced direction change after collision
            self.is_stuck() or self.forced_direction_change or self.detect_train_collisions(),
           
            # c. Wall danger (within 3 cells)
            self.check_wall_danger(steps=self.wall_buffer),
           
            # d. Train collision danger (immediate next cell)
            (grid_x + current_dir[0], grid_y + current_dir[1]) in occupied_positions
        ]
       
        if any(critical_conditions):
            # Reset forced direction flag if it was set
            self.forced_direction_change = False
           
            # Get best move in critical situation
            critical_move = self.get_best_move(grid_x, grid_y, current_dir, occupied_positions, target_pos)
            if critical_move:
                self.logger.debug(f"Train {self.nickname} fait un mouvement critique: {critical_move}")
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
            if move.value != (-current_dir[0], -current_dir[1]):  # Avoid U-turns
                return move
               
        # Ultimate fallback
        return random.choice([move for move in Move if move.value != (-current_dir[0], -current_dir[1])])
    
    # ------------------------------------------------------------------------------------------------------