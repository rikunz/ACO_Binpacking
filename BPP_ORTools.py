from ortools.linear_solver import pywraplp
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class BinPackingORTools:
    def __init__(self, items, bin_capacity):
        self.items = items
        self.bin_capacity = bin_capacity
        self.num_items = len(items)
        # Maximum bins needed worst case is number of items
        self.max_bins = self.num_items  
        
    def solve(self):
        """
        Solve bin packing using OR-Tools MIP solver
        Returns:
          - Number of bins used
          - List of bin weights
          - Solution time
        """
        start_time = time.time()
        
        # Create the MIP solver with CBC backend
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            return None

        # Variables
        # x[i,j] = 1 if item i is packed in bin j
        x = {}
        for i in range(self.num_items):
            for j in range(self.max_bins):
                x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')

        # y[j] = 1 if bin j is used
        y = {}
        for j in range(self.max_bins):
            y[j] = solver.IntVar(0, 1, f'y_{j}')

        # Constraints
        # Each item must be assigned to exactly one bin
        for i in range(self.num_items):
            solver.Add(sum(x[i,j] for j in range(self.max_bins)) == 1)

        # Bin capacity constraints 
        for j in range(self.max_bins):
            solver.Add(
                sum(self.items[i] * x[i,j] for i in range(self.num_items)) 
                <= self.bin_capacity * y[j])

        # Objective: minimize number of bins used
        solver.Minimize(sum(y[j] for j in range(self.max_bins)))

        # Solve
        status = solver.Solve()
        solve_time = time.time() - start_time

        if status == pywraplp.Solver.OPTIMAL:
            # Get number of bins used
            num_bins = sum(1 for j in range(self.max_bins) 
                         if y[j].solution_value() > 0.5)

            # Get bin weights
            bin_weights = []
            bin_contents = []
            
            for j in range(self.max_bins):
                if y[j].solution_value() > 0.5:
                    # Calculate bin weight
                    weight = sum(self.items[i] * x[i,j].solution_value() 
                               for i in range(self.num_items))
                    bin_weights.append(weight)
                    
                    # Get items in this bin
                    items_in_bin = []
                    for i in range(self.num_items):
                        if x[i,j].solution_value() > 0.5:
                            items_in_bin.append(self.items[i])
                    bin_contents.append(items_in_bin)

            return {
                "status": "OPTIMAL",
                "num_bins": num_bins,
                "bin_weights": bin_weights,
                "bin_contents": bin_contents,
                "solve_time": solve_time
            }
        else:
            return {
                "status": "FAILED",
                "solve_time": solve_time
            }
    
    def visualize_solution(self, result):
        """
        Visualize the bin packing solution using matplotlib.
        
        Args:
            result (dict): The result dictionary from the solve method
        """
        if result.get("status") != "OPTIMAL":
            print("No solution to visualize.")
            return

        bin_contents = result['bin_contents']
        bin_capacity = self.bin_capacity

        # Create the visualization
        fig, ax = plt.subplots(figsize=(15, 8))
        max_height = bin_capacity
        bin_width = 1
        spacing = 0.2

        for bin_idx, bin_items in enumerate(bin_contents):
            current_height = 0
            bin_x = bin_idx * (bin_width + spacing)

            # Draw bin outline
            ax.add_patch(Rectangle((bin_x, 0), bin_width, bin_capacity,
                                fill=False, color='black', linewidth=2))

            for item in bin_items:
                item_height = item
                # Use a color gradient or different colors for each item
                color = plt.cm.viridis(item_height / max(bin_items))
                ax.add_patch(Rectangle((bin_x, current_height), bin_width, item_height,
                                    fill=True, alpha=0.7, edgecolor='black', linewidth=1))

                # Add item weight text
                ax.text(bin_x + bin_width / 2, current_height + item_height / 2,
                        f'{item}',
                        horizontalalignment='center', verticalalignment='center',
                        fontweight='bold')

                current_height += item_height

        # Adjust plot
        total_bins = len(bin_contents)
        plt.xlim(-spacing, total_bins * (bin_width + spacing))
        plt.ylim(0, max_height * 1.1)
        plt.xticks([i * (bin_width + spacing) + bin_width / 2 for i in range(total_bins)],
                [f'{i + 1}' for i in range(total_bins)], rotation=45)
        plt.xlabel('Bins', fontsize=12)
        plt.ylabel('Bin Capacity', fontsize=12)
        plt.title('Bin Packing Solution Visualization using ORTools', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Input data
    items = [98, 96, 94, 92, 90, 88, 86, 84, 
    82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 
    58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 
    34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12]

    bin_capacity = 100

    # Solve using OR-Tools
    solver = BinPackingORTools(items, bin_capacity)
    result = solver.solve()

    # Print results
    print("\nOR-Tools Solution:")
    print("-" * 50)
    if result["status"] == "OPTIMAL":
        print(f"Status: {result['status']}")
        print(f"Number of bins used: {result['num_bins']}")
        print("\nBin contents:")
        for i, contents in enumerate(result['bin_contents']):
            print(f"Bin {i+1}: {contents} (sum = {sum(contents)})")
        print(f"\nSolve time: {result['solve_time']:.2f} seconds")
        
        # Visualize the solution
        solver.visualize_solution(result)
    else:
        print(f"Status: {result['status']}")
        print(f"Solve time: {result['solve_time']:.2f} seconds")