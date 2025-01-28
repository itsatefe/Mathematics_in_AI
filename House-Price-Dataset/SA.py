import numpy as np
import random
import math

class SimulatedAnnealing:
    def __init__(self, objective_function, initial_solution, max_iter=1000, initial_temp=1.0, cooling_rate=0.99):
        self.objective_function = objective_function
        self.solution = initial_solution.copy()
        self.best_solution = initial_solution.copy()
        self.max_iter = max_iter
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def minimize(self):
 
        current_solution = self.solution.copy()
        best_solution = self.solution.copy()

        current_cost = self.objective_function(current_solution)
        best_cost = current_cost

        temperature = self.initial_temp
        n_features = len(self.solution)

        for iteration in range(self.max_iter):
            new_solution = current_solution.copy()
            random_index = random.randint(0, n_features - 1)
            new_solution[random_index] += np.random.uniform(-1.0, 1.0)

            new_cost = self.objective_function(new_solution)

            if new_cost < current_cost:
                current_solution = new_solution
                current_cost = new_cost
                if new_cost < best_cost:
                    best_solution = new_solution.copy()
                    best_cost = new_cost
            else:
                acceptance_prob = math.exp(-(new_cost - current_cost) / temperature)
                if random.random() < acceptance_prob:
                    current_solution = new_solution
                    current_cost = new_cost

            temperature *= self.cooling_rate

            # Print progress
#             if iteration % 100 == 0:
#                 print(f"Iteration {iteration}: Best Cost = {best_cost}, Temperature = {temperature:.4f}")

        return best_solution, best_cost

