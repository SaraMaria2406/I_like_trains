import random
from common.base_agent import BaseAgent
from common.move import Move

# Student scipers, will be automatically used to evaluate your code
SCIPERS = ["359328", "391236"]


class Agent(BaseAgent):

    def __init__(self, nickname: str, network, logger: str = "client.agent"):
        super().__init__(nickname, network, logger)
        self.wall_buffer = 3  # Augmenté pour une meilleure anticipation
        self.last_positions = []  # Pour détecter les boucles
        self.max_positions_memory = 10
        self.stuck_threshold = 5  # Nombre de positions identiques avant de considérer être bloqué
        self.passenger_group_bonus = 2  # Bonus pour les groupes de passagers
        self.prev_trains = {}  # Pour détecter les trains morts
        self.forced_direction_change = False  # Pour forcer un changement de direction après collision

    def is_stuck(self):
        """Détecte si le train tourne en rond ou est bloqué"""
        if len(self.last_positions) < self.stuck_threshold:
            return False
            
        # Vérifier si on répète les mêmes positions
        unique_positions = set(self.last_positions[-self.stuck_threshold:])
        return len(unique_positions) <= 2  # Si on alterne entre 1 ou 2 positions

    def update_position_history(self):
        """Met à jour l'historique des positions du train"""
        if self.nickname not in self.all_trains:
            return
            
        current_pos = tuple(self.all_trains[self.nickname]["position"])
        self.last_positions.append(current_pos)
        if len(self.last_positions) > self.max_positions_memory:
            self.last_positions.pop(0)

    def get_passenger_clusters(self):
        """Identifie et évalue les groupes de passagers sur la carte"""
        if not self.passengers:
            return []
            
        # Convertir positions des passagers en coordonnées de grille
        grid_passengers = []
        for p in self.passengers:
            px, py = p["position"]
            grid_passengers.append((px // self.cell_size, py // self.cell_size))
        
        # Évaluer la densité des passagers (combien de passagers dans un rayon de 2 cases)
        clusters = []
        for i, (px, py) in enumerate(grid_passengers):
            nearby_count = 0
            for j, (qx, qy) in enumerate(grid_passengers):
                if i != j and abs(px - qx) + abs(py - qy) <= 2:  # Manhattan distance ≤ 2
                    nearby_count += 1
            
            clusters.append({
                "position": (px, py),
                "value": 1 + nearby_count * self.passenger_group_bonus  # Valeur du passager + bonus pour les voisins
            })
        
        return sorted(clusters, key=lambda x: x["value"], reverse=True)

    def is_path_safe(self, from_pos, to_pos):
        """Vérifie si le chemin entre deux positions est sûr (pas d'obstacles)"""
        from_x, from_y = from_pos
        to_x, to_y = to_pos
        
        # Identifier les points intermédiaires sur le chemin
        path_points = []
        
        # Si on se déplace horizontalement
        if from_y == to_y:
            step = 1 if from_x < to_x else -1
            for x in range(from_x + step, to_x + step, step):
                path_points.append((x, from_y))
        # Si on se déplace verticalement
        elif from_x == to_x:
            step = 1 if from_y < to_y else -1
            for y in range(from_y + step, to_y + step, step):
                path_points.append((from_x, y))
                
        # Vérifier les obstacles sur le chemin
        occupied_positions = set()
        for _, train in self.all_trains.items():
            if not train["alive"]:
                continue
                
            # Positions des têtes de train et des wagons
            train_x, train_y = train["position"]
            occupied_positions.add((train_x // self.cell_size, train_y // self.cell_size))
            
            for wagon in train["wagons"]:
                w_x, w_y = wagon
                occupied_positions.add((w_x // self.cell_size, w_y // self.cell_size))
        
        # Vérifier si un point du chemin est occupé
        for point in path_points:
            if point in occupied_positions:
                return False
                
        return True

    def move_to_best_passenger_cluster(self):
        """Se déplace vers le meilleur groupe de passagers en tenant compte de la distance et de la densité"""
        if not self.passengers or self.nickname not in self.all_trains:
            return None
        
        # Position actuelle en coordonnées de grille
        my_pos = self.all_trains[self.nickname]["position"]
        my_x = my_pos[0] // self.cell_size
        my_y = my_pos[1] // self.cell_size
        current_grid_pos = (my_x, my_y)
        
        # Obtenir les clusters de passagers
        clusters = self.get_passenger_clusters()
        if not clusters:
            return None
            
        # Évaluer chaque cluster en fonction de sa valeur et de la distance
        best_score = -float("inf")
        best_target = None
        
        for cluster in clusters:
            px, py = cluster["position"]
            dist = abs(px - my_x) + abs(py - my_y)  # Manhattan distance
            
            # Éviter division par zéro
            if dist == 0:
                continue
                
            # Score = valeur du cluster / distance (privilégie les clusters proches et denses)
            score = cluster["value"] / dist
            
            # Vérifier si le chemin est sûr
            if self.is_path_safe(current_grid_pos, (px, py)):
                score *= 1.5  # Bonus pour les chemins sûrs
            else:
                score *= 0.5  # Malus pour les chemins risqués
                
            if score > best_score:
                best_score = score
                best_target = (px, py)
        
        if best_target is None:
            # Fallback sur l'ancienne méthode si aucun cluster viable n'est trouvé
            return self.move_to_closest_passenger()
            
        px, py = best_target
        dx = px - my_x
        dy = py - my_y
        
        # Choisir la direction prioritaire (horizontal puis vertical)
        if dx != 0:
            move = Move.RIGHT if dx > 0 else Move.LEFT
        elif dy != 0:
            move = Move.DOWN if dy > 0 else Move.UP
        else:
            return None  # Déjà au bon endroit
            
        self.logger.debug(f"Train {self.nickname} heading toward passenger cluster at {best_target}")
        
        # Valider que le mouvement est dans les limites
        test_x = my_x + move.value[0]
        test_y = my_y + move.value[1]
        if (0 <= test_x < self.game_width // self.cell_size and 
            0 <= test_y < self.game_height // self.cell_size):
            return move
            
        return None

    def move_to_closest_passenger(self):
        """Version originale de la fonction pour le fallback"""
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
        """Navigate to the delivery zone when carrying passengers.
        Stay in the delivery zone until all passengers are delivered.
        Returns a Move if heading to or staying in delivery zone, None otherwise."""
    
        # Check if we exist and if delivery_zone exists
        if self.nickname not in self.all_trains or not hasattr(self, 'delivery_zone'):
            return None

        # Get this train's position
        my_train_data = self.all_trains[self.nickname]
        my_train_position = my_train_data["position"]

        # Check how many passengers this train is currently carrying
        onboard_count = len(self.all_trains[self.nickname]["wagons"])
    
        # Si on transporte suffisamment de passagers (3+), on va à la zone de livraison
        should_deliver = onboard_count >= 3
    
        # Si on est déjà dans le processus de livraison (dans ou en route vers la zone), 
        # on continue jusqu'à être vide
        delivery_pos = self.delivery_zone["position"]
        delivery_height = self.delivery_zone["height"]
        delivery_width = self.delivery_zone["width"]
    
        # Calculate the boundaries of the delivery zone in grid coordinates
        delivery_left = delivery_pos[0] // self.cell_size
        delivery_right = (delivery_pos[0] + delivery_width) // self.cell_size
        delivery_top = delivery_pos[1] // self.cell_size
        delivery_bottom = (delivery_pos[1] + delivery_height) // self.cell_size

        # Get train position in grid coordinates
        my_x = my_train_position[0] // self.cell_size
        my_y = my_train_position[1] // self.cell_size

        # Check if we're already inside the delivery zone
        inside_delivery_zone = (
            delivery_left <= my_x <= delivery_right and
            delivery_top <= my_y <= delivery_bottom
        )
        # Si on a des passagers ET qu'on est dans la zone de livraison, on reste
        if onboard_count > 0 and inside_delivery_zone:
            should_deliver = True
            self.logger.debug(f"Train {self.nickname} is inside delivery zone with {onboard_count} passengers, staying until empty.")
    
        if not should_deliver:
            # Si on n'a pas besoin de livrer (moins de 3 passagers et pas déjà dans la zone), 
            # on va chercher d'autres passagers
            return None
        
        self.logger.debug(f"Train {self.nickname} has {onboard_count} passengers and is heading to/staying in the delivery zone.")

        # Calculate the center of the delivery zone in grid coordinates
        delivery_center_x = (delivery_pos[0] + delivery_width // 2) // self.cell_size
        delivery_center_y = (delivery_pos[1] + delivery_height // 2) // self.cell_size
    
        # Calculate the boundaries of the delivery zone in grid coordinates
        delivery_left = delivery_pos[0] // self.cell_size
        delivery_right = (delivery_pos[0] + delivery_width) // self.cell_size
        delivery_top = delivery_pos[1] // self.cell_size
        delivery_bottom = (delivery_pos[1] + delivery_height) // self.cell_size
    
        # Check if we're already inside the delivery zone
        inside_delivery_zone = (
            delivery_left <= my_x <= delivery_right and
            delivery_top <= my_y <= delivery_bottom
        )
    
        if inside_delivery_zone:
            # Si on est déjà dans la zone de livraison, faire des allers-retours à l'intérieur
            # pour maximiser les chances de livrer tous les passagers
            self.logger.debug(f"Train {self.nickname} is inside delivery zone with {onboard_count} passengers, circulating.")
        
            # Récupérer la direction actuelle
            current_dir = my_train_data["direction"]
            dx, dy = current_dir
        
            # Calculer la prochaine position si on continue dans la même direction
            next_x = my_x + dx
            next_y = my_y + dy
        
            # Vérifier si on reste dans la zone de livraison en continuant tout droit
            if delivery_left <= next_x <= delivery_right and delivery_top <= next_y <= delivery_bottom:
                # Continuer dans la même direction
                for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    if move.value == current_dir:
                        return move
        
            # Si on va sortir de la zone, changer de direction pour rester dedans
            possible_moves = []
        
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                new_dx, new_dy = move.value
                next_x = my_x + new_dx
                next_y = my_y + new_dy
            
                # Ne pas faire de demi-tour (sauf si nécessaire)
                if (new_dx, new_dy) == (-dx, -dy):
                    continue
                
                # Vérifier si cette direction nous garde dans la zone de livraison
                if delivery_left <= next_x <= delivery_right and delivery_top <= next_y <= delivery_bottom:
                    possible_moves.append(move)
        
            # Si aucune autre direction ne nous garde dans la zone, accepter le demi-tour
            if not possible_moves:
                for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    new_dx, new_dy = move.value
                    next_x = my_x + new_dx
                    next_y = my_y + new_dy
                
                    if delivery_left <= next_x <= delivery_right and delivery_top <= next_y <= delivery_bottom:
                        possible_moves.append(move)
        
            # Choisir une direction qui nous garde dans la zone
            if possible_moves:
                return random.choice(possible_moves)
    
        # Si on n'est pas dans la zone de livraison, s'y diriger
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

        # Validate move is within bounds and safe from collisions
        test_x = my_x + move.value[0]
        test_y = my_y + move.value[1]
    
        # Position cible en coordonnées de grille
        target_pos = (test_x, test_y)
    
        # Vérifier si la cible est occupée par un train
        is_occupied = False
        for _, train in self.all_trains.items():
            if not train["alive"]:
                continue
            
            train_x, train_y = train["position"]
            train_grid_pos = (train_x // self.cell_size, train_y // self.cell_size)
        
            if train_grid_pos == target_pos:
                is_occupied = True
                break
            
            # Vérifier aussi les wagons
            for wagon in train["wagons"]:
                w_x, w_y = wagon
                wagon_grid_pos = (w_x // self.cell_size, w_y // self.cell_size)
            
                if wagon_grid_pos == target_pos:
                    is_occupied = True
                    break
    
        if (0 <= test_x < self.game_width // self.cell_size and
            0 <= test_y < self.game_height // self.cell_size and
            not is_occupied):
            return move

        # Si la direction principale n'est pas sûre, essayer l'autre axe
        if dx != 0 and dy != 0:
            # Essayer l'axe vertical si l'horizontal est bloqué
            alt_move = Move.DOWN if dy > 0 else Move.UP
            test_x = my_x + alt_move.value[0]
            test_y = my_y + alt_move.value[1]
        
            if (0 <= test_x < self.game_width // self.cell_size and
                0 <= test_y < self.game_height // self.cell_size):
                return alt_move
    
        return None

    def predict_train_positions(self):
        """Prédit les positions futures des autres trains basées sur leurs directions actuelles"""
        predicted_positions = set()
        
        for name, train in self.all_trains.items():
            if not train["alive"] or name == self.nickname:
                continue
                
            # Position et direction actuelles
            pos_x, pos_y = train["position"]
            dir_x, dir_y = train["direction"]
            
            # Prédire les 3 prochaines positions potentielles
            for steps in range(1, 4):
                future_x = pos_x + (dir_x * self.cell_size * steps)
                future_y = pos_y + (dir_y * self.cell_size * steps)
                
                # Convertir en coordonnées de grille
                grid_x = future_x // self.cell_size
                grid_y = future_y // self.cell_size
                
                predicted_positions.add((grid_x, grid_y))
        
        return predicted_positions

    def avoiding_trains(self):  
        """
        Returns a direction to avoid crashing into other trains.
        Enhanced to predict future positions of other trains.
        """
        if self.nickname not in self.all_trains:
            return None
            
        my_train = self.all_trains[self.nickname]
        if not my_train["alive"]:
            return None
            
        current_x, current_y = my_train["position"]
        dir_x, dir_y = my_train["direction"]
        
        # Convertir en coordonnées de grille
        grid_x = current_x // self.cell_size
        grid_y = current_y // self.cell_size

        # Positions actuelles des trains et wagons
        occupied_positions = set()
        
        for name, train in self.all_trains.items():
            if not train["alive"]:
                continue

            train_x, train_y = train["position"]
            occupied_positions.add((train_x // self.cell_size, train_y // self.cell_size))
            
            for wagon in train["wagons"]:
                w_x, w_y = wagon
                occupied_positions.add((w_x // self.cell_size, w_y // self.cell_size))
        
        # Ajouter les positions futures prédites
        predicted_positions = self.predict_train_positions()
        occupied_positions.update(predicted_positions)

        # Calculer la prochaine position si on continue tout droit
        next_grid_x = grid_x + dir_x
        next_grid_y = grid_y + dir_y
        next_pos = (next_grid_x, next_grid_y)

        # Si on va percuter un train ou entrer dans une zone à risque
        if next_pos in occupied_positions:
            safe_directions = []
            direction_scores = {}

            for move_option in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dx, dy = move_option.value
                new_grid_x = grid_x + dx
                new_grid_y = grid_y + dy
                new_pos = (new_grid_x, new_grid_y)

                # Ignorer demi-tour et position actuelle
                if (dx, dy) == (dir_x, dir_y) or (dx, dy) == (-dir_x, -dir_y):
                    continue

                # Vérifier les limites de la carte
                if (0 <= new_grid_x < self.game_width // self.cell_size and 
                    0 <= new_grid_y < self.game_height // self.cell_size):
                    
                    # Vérifier si la position est libre
                    if new_pos not in occupied_positions:
                        safe_directions.append(move_option)
                        
                        # Calculer un score pour cette direction
                        # Favoriser les directions qui vont vers l'objectif actuel
                        score = 1  # Score de base
                        
                        # Si on transporte des passagers, favoriser la direction vers la zone de livraison
                        if len(my_train["wagons"]) >= 3 and hasattr(self, 'delivery_zone'):
                            delivery_pos = self.delivery_zone["position"]
                            delivery_center_x = (delivery_pos[0] + self.delivery_zone["width"] // 2) // self.cell_size
                            delivery_center_y = (delivery_pos[1] + self.delivery_zone["height"] // 2) // self.cell_size
                            
                            # Distance après le mouvement
                            new_distance = abs(new_grid_x - delivery_center_x) + abs(new_grid_y - delivery_center_y)
                            current_distance = abs(grid_x - delivery_center_x) + abs(grid_y - delivery_center_y)
                            
                            if new_distance < current_distance:
                                score += 2  # Bonus si on se rapproche de la zone de livraison
                        
                        # Si on cherche des passagers, favoriser la direction vers des passagers
                        elif self.passengers:
                            # Trouver le passager le plus proche
                            best_passenger_pos = None
                            min_dist = float("inf")
                            
                            for p in self.passengers:
                                px, py = p["position"]
                                px //= self.cell_size
                                py //= self.cell_size
                                dist = abs(px - grid_x) + abs(py - grid_y)
                                
                                if dist < min_dist:
                                    min_dist = dist
                                    best_passenger_pos = (px, py)
                            
                            if best_passenger_pos:
                                px, py = best_passenger_pos
                                # Distance après le mouvement
                                new_distance = abs(new_grid_x - px) + abs(new_grid_y - py)
                                current_distance = abs(grid_x - px) + abs(grid_y - py)
                                
                                if new_distance < current_distance:
                                    score += 2  # Bonus si on se rapproche du passager
                        
                        direction_scores[move_option] = score
            
            if safe_directions:
                # Choisir la direction avec le meilleur score
                if direction_scores:
                    best_directions = [d for d, s in direction_scores.items() 
                                     if s == max(direction_scores.values())]
                    return random.choice(best_directions)
                else:
                    return random.choice(safe_directions)
            else:
                # Si bloqué de tous côtés, tenter un demi-tour en dernier recours
                reverse_x, reverse_y = -dir_x, -dir_y
                reverse_pos = (grid_x + reverse_x, grid_y + reverse_y)
                
                if (0 <= reverse_pos[0] < self.game_width // self.cell_size and 
                    0 <= reverse_pos[1] < self.game_height // self.cell_size and
                    reverse_pos not in occupied_positions):
                    
                    for move_option in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                        if move_option.value == (reverse_x, reverse_y):
                            self.logger.debug(f"Train {self.nickname} making emergency reverse!")
                            return move_option

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
        
        # Convertir en coordonnées de grille
        grid_x = current_x // self.cell_size
        grid_y = current_y // self.cell_size
    
        # Check several steps ahead for wall detection (using wall_buffer)
        for steps in range(1, self.wall_buffer + 1):
            next_grid_x = grid_x + (dir_x * steps)
            next_grid_y = grid_y + (dir_y * steps)
            
            # Si on approche d'un mur, changer de direction
            if (next_grid_x < 0 or next_grid_x >= self.game_width // self.cell_size or
                next_grid_y < 0 or next_grid_y >= self.game_height // self.cell_size):
                
                self.logger.debug(f"Train {self.nickname} avoiding wall at {grid_x},{grid_y}, buffer={steps}")
                
                # Calculer les directions sûres
                available_directions = []
                direction_scores = {}
                
                # Vérifier chaque direction potentielle
                for move_option in [Move.RIGHT, Move.LEFT, Move.UP, Move.DOWN]:
                    # Ne pas considérer la direction actuelle ou son opposé
                    if move_option.value == (dir_x, dir_y) or move_option.value == (-dir_x, -dir_y):
                        continue
                    
                    # Vérifier si cette direction mène à un mur
                    test_x = grid_x + move_option.value[0]
                    test_y = grid_y + move_option.value[1]
                    
                    if (0 <= test_x < self.game_width // self.cell_size and 
                        0 <= test_y < self.game_height // self.cell_size):
                        
                        available_directions.append(move_option)
                        
                        # Calculer un score pour cette direction
                        score = 1  # Score de base
                        
                        # Bonus pour les directions qui offrent plus d'espace
                        space_score = 0
                        for look_ahead in range(1, 4):
                            look_x = test_x + move_option.value[0] * look_ahead
                            look_y = test_y + move_option.value[1] * look_ahead
                            
                            if (0 <= look_x < self.game_width // self.cell_size and 
                                0 <= look_y < self.game_height // self.cell_size):
                                space_score += 1
                        
                        score += space_score
                        
                        # Si on transporte des passagers, favoriser la direction vers la zone de livraison
                        if len(my_train["wagons"]) >= 3 and hasattr(self, 'delivery_zone'):
                            delivery_pos = self.delivery_zone["position"]
                            delivery_center_x = (delivery_pos[0] + self.delivery_zone["width"] // 2) // self.cell_size
                            delivery_center_y = (delivery_pos[1] + self.delivery_zone["height"] // 2) // self.cell_size
                            
                            # Distance après le mouvement
                            new_distance = abs(test_x - delivery_center_x) + abs(test_y - delivery_center_y)
                            current_distance = abs(grid_x - delivery_center_x) + abs(grid_y - delivery_center_y)
                            
                            if new_distance < current_distance:
                                score += 3  # Bonus significatif si on se rapproche de la zone de livraison
                        
                        direction_scores[move_option] = score
                
                # Choisir une direction sûre
                if direction_scores:
                    best_directions = [d for d, s in direction_scores.items() 
                                     if s == max(direction_scores.values())]
                    return random.choice(best_directions)
                elif available_directions:
                    return random.choice(available_directions)
                
                # Aucune direction valide trouvée
                self.logger.debug(f"Train {self.nickname} has no valid directions!")
        
        # Pas de mur détecté dans notre buffer
        return None

    def detect_train_collisions(self):
        """
        Détecte si un train est mort à notre position ou près de nous,
        ce qui pourrait indiquer une collision récente.
        """
        if not hasattr(self, 'prev_trains') or not self.all_trains or self.nickname not in self.all_trains:
            self.prev_trains = {name: train["alive"] for name, train in self.all_trains.items()}
            return False
            
        my_pos = self.all_trains[self.nickname]["position"]
        my_grid_x = my_pos[0] // self.cell_size
        my_grid_y = my_pos[1] // self.cell_size
        
        collision_detected = False
        
        # Vérifier les trains qui étaient vivants avant mais qui sont morts maintenant
        for name, train in self.all_trains.items():
            if name != self.nickname and name in self.prev_trains and self.prev_trains[name] and not train["alive"]:
                # Ce train vient de mourir, vérifier s'il est près de nous
                dead_pos = train["position"]
                dead_grid_x = dead_pos[0] // self.cell_size
                dead_grid_y = dead_pos[1] // self.cell_size
                
                # Si le train mort est à notre position ou adjacent, c'est probablement une collision
                if abs(dead_grid_x - my_grid_x) <= 1 and abs(dead_grid_y - my_grid_y) <= 1:
                    self.logger.debug(f"Train {self.nickname} detected collision with now-dead train {name}!")
                    collision_detected = True
                    self.forced_direction_change = True
                    break
        
        # Mettre à jour la liste des trains pour la prochaine vérification
        self.prev_trains = {name: train["alive"] for name, train in self.all_trains.items()}
        return collision_detected

    def get_move(self):
        """
        Called regularly to get the next move for your train.
        Enhanced with better safety checks and decision-making.
        """
        # Mise à jour de l'historique des positions
        self.update_position_history()
        
        if self.nickname not in self.all_trains or not self.all_trains[self.nickname]["alive"]:
            return random.choice(list(Move))
    
        # Récupérer les informations sur notre train
        my_train = self.all_trains[self.nickname]
        current_pos = my_train["position"]
        current_dir = my_train["direction"]
        grid_x = current_pos[0] // self.cell_size
        grid_y = current_pos[1] // self.cell_size
        
        # Détecter si une collision vient de se produire
        collision_detected = self.detect_train_collisions()
    
        # Vérifier si on est bloqué dans une boucle ou si on a détecté une collision
        if self.is_stuck() or self.forced_direction_change or collision_detected:
            if self.is_stuck():
                self.logger.debug(f"Train {self.nickname} detected it is stuck in a loop. Breaking pattern.")
            if self.forced_direction_change or collision_detected:
                self.logger.debug(f"Train {self.nickname} forcing direction change after collision!")
                self.forced_direction_change = False  # Réinitialiser le flag
                
            # Trouver toutes les directions sûres (pas de murs, pas de trains)
            safe_directions = []
            occupied_positions = set()
            
            # Rassembler positions des trains et wagons
            for name, train in self.all_trains.items():
                if not train["alive"]:
                    continue
                
                train_x, train_y = train["position"]
                occupied_positions.add((train_x // self.cell_size, train_y // self.cell_size))
                
                for wagon in train["wagons"]:
                    w_x, w_y = wagon
                    occupied_positions.add((w_x // self.cell_size, w_y // self.cell_size))
            
            # Vérifier chaque direction
            for move_option in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                # Éviter la direction actuelle et son opposé pour vraiment changer de direction
                if collision_detected and (move_option.value == current_dir or move_option.value == (-current_dir[0], -current_dir[1])):
                    continue
                
                test_x = grid_x + move_option.value[0]
                test_y = grid_y + move_option.value[1]
                test_pos = (test_x, test_y)
                
                if (0 <= test_x < self.game_width // self.cell_size and 
                    0 <= test_y < self.game_height // self.cell_size and
                    test_pos not in occupied_positions):
                    
                    safe_directions.append(move_option)
            
            if safe_directions:
                chosen_direction = random.choice(safe_directions)
                self.logger.debug(f"Train {self.nickname} changing direction to {chosen_direction}")
                return chosen_direction
    
        # Vérifier le danger immédiat (1 case devant)
        next_grid_x = grid_x + current_dir[0]
        next_grid_y = grid_y + current_dir[1]
        
        # Vérifier si on sort des limites
        immediate_wall_danger = (next_grid_x < 0 or next_grid_x >= self.game_width // self.cell_size or
                                next_grid_y < 0 or next_grid_y >= self.game_height // self.cell_size)
    
        # Vérifier collision avec un autre train
        immediate_train_danger = False
        next_pos = (next_grid_x, next_grid_y)
        
        # Rassembler toutes les positions occupées
        occupied_positions = set()
        for name, train in self.all_trains.items():
            if not train["alive"]:
                continue
                
            train_x, train_y = train["position"]
            occupied_positions.add((train_x // self.cell_size, train_y // self.cell_size))
            
            for wagon in train["wagons"]:
                w_x, w_y = wagon
                occupied_positions.add((w_x // self.cell_size, w_y // self.cell_size))
        
        immediate_train_danger = next_pos in occupied_positions
    
        # DANGER CRITIQUE: Éviter les murs en premier
        if immediate_wall_danger:
            new_direction = self.avoiding_walls()
            if new_direction:
                self.logger.debug(f"Train {self.nickname} in CRITICAL DANGER - emergency wall avoidance {new_direction}")
                return new_direction
    
        # DANGER ÉLEVÉ: Éviter les autres trains en danger imminent
        if immediate_train_danger:
            new_direction = self.avoiding_trains()
            if new_direction:
                self.logger.debug(f"Train {self.nickname} in HIGH DANGER - emergency train avoidance {new_direction}")
                return new_direction
    
        # Si on transporte suffisamment de passagers, priorité à la livraison
        # (cette condition est vérifiée dans go_to_delivery_zone)
        to_delivery = self.go_to_delivery_zone()
        if to_delivery:
            # Vérifier la sécurité du mouvement proposé
            test_grid_x = grid_x + to_delivery.value[0]
            test_grid_y = grid_y + to_delivery.value[1]
            test_pos = (test_grid_x, test_grid_y)
            
            if (0 <= test_grid_x < self.game_width // self.cell_size and 
                0 <= test_grid_y < self.game_height // self.cell_size and
                test_pos not in occupied_positions):
                
                self.logger.debug(f"Train {self.nickname} going to delivery zone with move {to_delivery}")
                return to_delivery
    
        # Si on n'est pas en train de livrer, on va chercher des passagers
        # Utiliser la nouvelle méthode de clustering des passagers
        to_passenger = self.move_to_best_passenger_cluster()
        if to_passenger:
            # Vérifier la sécurité du mouvement proposé
            test_grid_x = grid_x + to_passenger.value[0]
            test_grid_y = grid_y + to_passenger.value[1]
            test_pos = (test_grid_x, test_grid_y)
            
            if (0 <= test_grid_x < self.game_width // self.cell_size and 
                0 <= test_grid_y < self.game_height // self.cell_size and
                test_pos not in occupied_positions):
                
                self.logger.debug(f"Train {self.nickname} heading to passenger cluster with move {to_passenger}")
                return to_passenger
    
        # Évitement proactif des murs (anticipation)
        # Moins prioritaire que les actions principales mais important pour la sécurité
        new_direction = self.avoiding_walls()
        if new_direction:
            test_grid_x = grid_x + new_direction.value[0]
            test_grid_y = grid_y + new_direction.value[1]
            test_pos = (test_grid_x, test_grid_y)
            
            if (0 <= test_grid_x < self.game_width // self.cell_size and 
                0 <= test_grid_y < self.game_height // self.cell_size and
                test_pos not in occupied_positions):
                
                self.logger.debug(f"Train {self.nickname} proactively avoiding wall with move {new_direction}")
                return new_direction
        
        # Évitement proactif des trains (anticipation)
        new_direction = self.avoiding_trains()
        if new_direction:
            test_grid_x = grid_x + new_direction.value[0]
            test_grid_y = grid_y + new_direction.value[1]
            test_pos = (test_grid_x, test_grid_y)
            
            if (0 <= test_grid_x < self.game_width // self.cell_size and 
                0 <= test_grid_y < self.game_height // self.cell_size and
                test_pos not in occupied_positions):
                
                self.logger.debug(f"Train {self.nickname} proactively avoiding other trains with move {new_direction}")
                return new_direction
    
        # Si le train est bloqué ou ne peut pas avancer en sécurité, chercher une direction alternative
        if immediate_wall_danger or immediate_train_danger:
            safe_directions = []
            
            for move_option in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                # Ne pas considérer le demi-tour
                if move_option.value == (-current_dir[0], -current_dir[1]):
                    continue
                    
                test_grid_x = grid_x + move_option.value[0]
                test_grid_y = grid_y + move_option.value[1]
                test_pos = (test_grid_x, test_grid_y)
                
                if (0 <= test_grid_x < self.game_width // self.cell_size and 
                    0 <= test_grid_y < self.game_height // self.cell_size and
                    test_pos not in occupied_positions):
                    
                    safe_directions.append(move_option)
            
            if safe_directions:
                random_safe_move = random.choice(safe_directions)
                self.logger.debug(f"Train {self.nickname} choosing random safe direction {random_safe_move}")
                return random_safe_move
            else:
                # En dernier recours, tenter un demi-tour si toutes les directions sont bloquées
                reverse_dir = None
                for move_option in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    if move_option.value == (-current_dir[0], -current_dir[1]):
                        reverse_dir = move_option
                        break
                
                if reverse_dir:
                    test_grid_x = grid_x + reverse_dir.value[0]
                    test_grid_y = grid_y + reverse_dir.value[1]
                    test_pos = (test_grid_x, test_grid_y)
                    
                    if (0 <= test_grid_x < self.game_width // self.cell_size and 
                        0 <= test_grid_y < self.game_height // self.cell_size and
                        test_pos not in occupied_positions):
                        
                        self.logger.debug(f"Train {self.nickname} making emergency reversal to {reverse_dir}")
                        return reverse_dir
    
        # Si aucune action spéciale n'est nécessaire, maintenir la direction actuelle
        # si elle est sûre
        current_move = None
        for m in Move:
            if m.value == current_dir:
                current_move = m
                break
        
        if current_move:
            next_grid_x = grid_x + current_move.value[0]
            next_grid_y = grid_y + current_move.value[1]
            next_pos = (next_grid_x, next_grid_y)
            
            # Vérifier que la position suivante est sûre
            if (0 <= next_grid_x < self.game_width // self.cell_size and 
                0 <= next_grid_y < self.game_height // self.cell_size and
                next_pos not in occupied_positions):
                
                return current_move
        
        # Si la direction actuelle n'est pas sûre, choisir une direction aléatoire mais sûre
        safe_directions = []
        for move_option in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            test_grid_x = grid_x + move_option.value[0]
            test_grid_y = grid_y + move_option.value[1]
            test_pos = (test_grid_x, test_grid_y)
            
            if (0 <= test_grid_x < self.game_width // self.cell_size and 
                0 <= test_grid_y < self.game_height // self.cell_size and
                test_pos not in occupied_positions):
                
                safe_directions.append(move_option)
        
        if safe_directions:
            random_direction = random.choice(safe_directions)
            self.logger.debug(f"Train {self.nickname} choosing random safe direction {random_direction}")
            return random_direction
        
        # Fallback - en dernier recours, maintenir la direction actuelle même si elle n'est pas idéale
        for m in Move:
            if m.value == current_dir:
                return m
        
        # Si tout échoue, choisir une direction aléatoire
        return random.choice(list(Move))