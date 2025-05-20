import random
from common.base_agent import BaseAgent
from common.move import Move

# Student scipers, will be automatically used to evaluate your code
SCIPERS = ["359328", "391236"]

# all_trains: {'Player1': {'position': [380, 260], 'wagons': [], 'direction': [0, -1], 'score': 0, 'color': [122, 223, 95], 'alive': True, 'boost_cooldown_active': False}}

class Agent(BaseAgent):

    def __init__(self, nickname: str, network, logger: str = "client.agent"):
        super().__init__(nickname, network, logger)  # Ajoute dans BaseAgent() ? --> chercher sur google
        self.wall_buffer = 2


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

        # Calcul des positions occupées (autres trains)
        occupied_positions = set()
        for name, train in self.all_trains.items():
            if not train["alive"]:
                continue
            occupied_positions.add(tuple(train["position"]))
            for wagon in train["wagons"]:
                occupied_positions.add(tuple(wagon))
   
        # Check several steps ahead for wall detection (using wall_buffer)
        for steps in range(1, self.wall_buffer + 1):
            next_x = current_x + (dir_x * self.cell_size * steps)
            next_y = current_y + (dir_y * self.cell_size * steps)
           
            # If we're approaching a wall, we need to change direction
            if (next_x < 0 or next_x >= self.game_width or
                next_y < 0 or next_y >= self.game_height):
               
                self.logger.debug(f"[{self.nickname}] Danger: mur détecté à ({next_x},{next_y})")
                self.logger.debug(f"Train {self.nickname} avoiding wall at {current_x},{current_y}")
               
                # Calculate available safe directions
                safe_directions = []
               
                # Check each potential direction to see if it's safe
                for move_option in [Move.RIGHT, Move.LEFT, Move.UP, Move.DOWN]:
                    
                    dx, dy = move_option.value

                # Ignore direction actuelle et demi-tour
                    if (dx, dy) == (dir_x, dir_y) or (dx, dy) == (-dir_x, -dir_y):
                        continue

                    new_x = current_x + dx * self.cell_size
                    new_y = current_y + dy * self.cell_size

                # Vérifie les murs
                    if not (0 <= new_x < self.game_width and 0 <= new_y < self.game_height):
                        continue

                # Vérifie collisions avec autres trains
                    if (new_x, new_y) in occupied_positions:
                        continue

                    safe_directions.append(move_option)

                if safe_directions:
                    chosen = random.choice(safe_directions)
                    self.logger.debug(f"[{self.nickname}] Nouvelle direction sûre: {chosen}")
                    return chosen
                else:
                    self.logger.debug(f"[{self.nickname}] Aucune direction sûre disponible pour éviter le mur.")

    # Aucun mur imminent ou pas de meilleure alternative
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

    def get_move(self):
       
        # First priority: go to passenger
        to_passenger = self.move_to_closest_passenger()
        if to_passenger:
            return to_passenger

        train = self.all_trains[self.nickname]
        current_x, current_y = train["position"]
        dir_x, dir_y = train["direction"]

        # Liste des positions occupées (trains adverses)
        occupied_positions = set()
        for name, t in self.all_trains.items():
            if not t["alive"]:
                continue
            occupied_positions.add(tuple(t["position"]))
            for wagon in t["wagons"]:
                occupied_positions.add(tuple(wagon))

        # --- Étape 1 : éviter les murs ---
        wall_avoidance = self.avoiding_walls()
        if wall_avoidance:
            dx, dy = wall_avoidance.value
            test_x = current_x + dx * self.cell_size
            test_y = current_y + dy * self.cell_size

            if (test_x, test_y) not in occupied_positions:
                self.logger.debug(f"[{self.nickname}] Évite mur → {wall_avoidance}")
                return wall_avoidance
            else:
                self.logger.debug(f"[{self.nickname}] Direction mur évitée mais occupée par train → {wall_avoidance}")

        # --- Étape 2 : éviter les trains ---
        train_avoidance = self.avoiding_trains()
        if train_avoidance:
            dx, dy = train_avoidance.value
            test_x = current_x + dx * self.cell_size
            test_y = current_y + dy * self.cell_size

            if (0 <= test_x < self.game_width and 0 <= test_y < self.game_height):
                self.logger.debug(f"[{self.nickname}] Évite train → {train_avoidance}")
                return train_avoidance

        # Étape 3 : aller vers le passager le plus proche si possible
        move_to_passenger = self.move_to_closest_passenger()
        if move_to_passenger:
            dx, dy = move_to_passenger.value
            test_x = current_x + dx * self.cell_size
            test_y = current_y + dy * self.cell_size
            if (0 <= test_x < self.game_width and 0 <= test_y < self.game_height and
                (test_x, test_y) not in occupied_positions):
                return move_to_passenger

        # --- Étape 4 : continuer tout droit si c’est safe ---
        next_x = current_x + dir_x * self.cell_size
        next_y = current_y + dir_y * self.cell_size
        if (0 <= next_x < self.game_width and 0 <= next_y < self.game_height and
            (next_x, next_y) not in occupied_positions):
            for m in Move:
                if m.value == (dir_x, dir_y):
                    self.logger.debug(f"[{self.nickname}] Continue tout droit")
                    return m

        # --- Étape 4 : aucune direction safe trouvée, choisir une au hasard ---
        self.logger.debug(f"[{self.nickname}] Aucune direction sûre, choix aléatoire de survie")
        return random.choice(list(Move))
       
        # Fallback: choose a random direction
        # return random.choice([Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT])

    
        
    
    
