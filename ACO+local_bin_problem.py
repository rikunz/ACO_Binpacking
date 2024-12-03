import random
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class AntColonyBinPacking:
    def __init__(self, items, bin_capacity, n_ants=None, beta=2, rho=0.5, k=2, max_iterations=200):
        """
        Initialize ACO for Bin Packing Problem
        """
        self.items = sorted(items, reverse=True)  # Sort items to improve packing
        self.bin_capacity = bin_capacity
        self.n_ants = n_ants or len(items)
        self.beta = beta
        self.rho = rho
        self.k = k
        self.max_iterations = max_iterations
        self.pheromones = [1.0] * len(self.items)

        # Properties to store the best solution
        self.best_solution = None
        self.best_num_bins = float('inf')
        self.best_bin_weights = None
        self.best_fitness = float('-inf')

    def heuristic(self, item, bin_weight):
        remaining_space = self.bin_capacity - (bin_weight + item)
        return 1 / (1 + max(0, remaining_space)) if remaining_space >= 0 else 0

    def fitness(self, bin_weights):
        bin_weights = self.clean_empty_bins(bin_weights)
        N = len(bin_weights)
        if N == 0:
            return float('-inf')  # Invalid solution
        fitness = sum((weight / self.bin_capacity) ** self.k for weight in bin_weights) / N
        return fitness

    def clean_empty_bins(self, bin_weights):
        """Remove bins with zero weight."""
        return [weight for weight in bin_weights if weight > 0]

    def construct_solution(self):
        """Construct a solution using the pheromone and heuristic information."""
        solution = [-1] * len(self.items)
        bin_weights = []

        for i, item in enumerate(self.items):
            bin_probabilities = self.calculate_bin_probabilities(item, bin_weights)
            selected_bin = self.select_bin(bin_probabilities)
            solution[i] = selected_bin

            if selected_bin == len(bin_weights):  # Create a new bin
                bin_weights.append(item)
            else:
                bin_weights[selected_bin] += item

        return solution

    def calculate_bin_probabilities(self, item, bin_weights):
        probabilities = []
        for j, bin_weight in enumerate(bin_weights):
            if bin_weight + item <= self.bin_capacity:
                prob = (self.pheromones[j] ** 1) * (self.heuristic(item, bin_weight) ** self.beta)
                probabilities.append((j, prob))
        probabilities.append((len(bin_weights), 1.0))
        return probabilities

    def select_bin(self, probabilities):
        total = sum(prob for _, prob in probabilities)
        r = random.uniform(0, total)
        cumulative = 0
        for bin_index, prob in probabilities:
            cumulative += prob
            if r <= cumulative:
                return bin_index
        return probabilities[-1][0]

    def update_pheromones(self, solution, bin_weights):
        self.pheromones = [p * (1 - self.rho) for p in self.pheromones]
        fitness_value = self.fitness(bin_weights)
        for i, bin_index in enumerate(solution):
            if bin_index < len(self.pheromones):
                self.pheromones[bin_index] += fitness_value

    def calculate_bin_weights(self, solution):
        bin_weights = []
        for i, bin_index in enumerate(solution):
            while bin_index >= len(bin_weights):
                bin_weights.append(0)
            bin_weights[bin_index] += self.items[i]
        return bin_weights

    def solve(self):
        """
        Run the ACO algorithm to solve the Bin Packing Problem.
        """
        for iteration in range(self.max_iterations):
            solution = self.construct_solution()
            bin_weights = self.calculate_bin_weights(solution)
            bin_weights = self.clean_empty_bins(bin_weights)
            num_bins = len(bin_weights)
            fitness_value = self.fitness(bin_weights)

            # Print fitness for the current iteration
            print(f"Iteration {iteration + 1}: Fitness = {fitness_value}")

            # Update best solution
            if fitness_value > self.best_fitness:  # Maximize fitness
                self.best_solution = self.group_items_by_bins(solution)
                self.best_num_bins = num_bins
                self.best_bin_weights = bin_weights
                self.best_fitness = fitness_value

            # Update pheromones
            self.update_pheromones(solution, bin_weights)

    def group_items_by_bins(self, solution):
        """Group items into bins based on the solution."""
        bins = [[] for _ in range(max(solution) + 1)]
        for item_idx, bin_idx in enumerate(solution):
            bins[bin_idx].append(item_idx)
        return bins

    def visualize_best_solution(self):
        """Visualize the best solution as packed bins."""
        if not self.best_solution:
            print("No solution to visualize.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        max_height = max(self.items) * 1.2
        bin_width = 1
        spacing = 0.2

        for bin_idx, bin_items in enumerate(self.best_solution):
            current_height = 0
            bin_x = bin_idx * (bin_width + spacing)

            # Draw bin outline
            ax.add_patch(Rectangle((bin_x, 0), bin_width, self.bin_capacity,
                                fill=False, color='black', linewidth=2))

            for item_idx in bin_items:
                item_height = self.items[item_idx]
                ax.add_patch(Rectangle((bin_x, current_height), bin_width, item_height,
                                    fill=True, alpha=0.5))

                # Add item weight text
                ax.text(bin_x + bin_width / 2, current_height + item_height / 2,
                        f'{self.items[item_idx]}',
                        horizontalalignment='center', verticalalignment='center')

                current_height += item_height

        # Adjust plot
        total_bins = len(self.best_solution)
        plt.xlim(-spacing, total_bins * (bin_width + spacing))
        plt.ylim(0, max(self.bin_capacity * 1.1, max_height))
        plt.xticks([i * (bin_width + spacing) + bin_width / 2 for i in range(total_bins)],
                [f'{i + 1}' for i in range(total_bins)], rotation=90)
        plt.xlabel('Bins')
        plt.ylabel('Capacity Used')
        plt.title('Best Solution Bin Packing Visualization')
        plt.grid(True)
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    items = [98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 
            68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 
            38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12]
    bin_capacity = 100
    aco = AntColonyBinPacking(items, bin_capacity=bin_capacity, n_ants=200, beta=2, 
                            rho=0.5, k=2, max_iterations=300)

    aco.solve()
    print("\nFinal Solution:")
    print({
        "num_bins": aco.best_num_bins,
        "bin_weights": aco.best_bin_weights,
        "fitness": aco.best_fitness
    })
    aco.visualize_best_solution()