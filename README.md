# Evolutionary-Computation_Assignment-3


    import random
    import operator
    import copy
    import matplotlib.pyplot as plt  # Used for plotting fitness over generations
    
    #  GP PARAMETERS 
    POP_SIZE = 50
    MIN_DEPTH = 2
    MAX_DEPTH = 4
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.2
    GENERATIONS = 50
    DESIRED_PRECISION = 0.001
    TOURNAMENT_SIZE = 3   
    ELITISM_SIZE = 1     
    
    # SAFE OPERATORS
    def safe_div(x, y):
        return x / y if abs(y) > 1e-6 else 1
    
    # FUNCTION & TERMINAL SET 
    FUNCTIONS = {
        'add': operator.add,
        'sub': operator.sub,
        'mul': operator.mul,
        'div': safe_div
    }
    TERMINALS = ['x', 'y', -3, -2, -1, 0, 1, 2, 3]
    # =============================================================================
    # 
    # =============================================================================
    #### Initialization
    #  TREE STRUCTURE 
    class GPTree:
        def init(self, data=None, left=None, right=None):
            self.data = data
            self.left = left
            self.right = right
    
        def node_label(self):
            return self.data.__name__ if callable(self.data) else str(self.data)
    
        def print_tree(self, indent='', last=True):
            branch = '`- ' if last else '|- '
            print(f"{indent}{branch}{self.node_label()}")
            indent += '   ' if last else '|  '
            children = []
            if self.left: children.append((self.left, False))
            if self.right: children.append((self.right, True))
            for child, is_last in children:
                child.print_tree(indent, is_last)
    
        def evaluate(self, x, y):
            if callable(self.data):
                return self.data(self.left.evaluate(x, y), self.right.evaluate(x, y))
            if self.data == 'x': return x
            if self.data == 'y': return y
            return self.data
    
        def size(self):
            return 1 + (self.left.size() if self.left else 0) + (self.right.size() if self.right else 0)
    
        def copy(self):
            return copy.deepcopy(self)
    
        def random_subtree(self, grow, max_depth, current_depth=0):
            if current_depth < MIN_DEPTH or (current_depth < max_depth and not grow):
                self.data = random.choice(list(FUNCTIONS.values()))
            elif current_depth >= max_depth:
                self.data = random.choice(TERMINALS)
            else:
                self.data = random.choice(list(FUNCTIONS.values()) + TERMINALS)
    
            if callable(self.data):
                self.left = GPTree()
                self.right = GPTree()
                self.left.random_subtree(grow, max_depth, current_depth + 1)
                self.right.random_subtree(grow, max_depth, current_depth + 1)
            else:
                self.left = None
                self.right = None
    
        def collect_nodes(self):
            nodes = [self]
            if self.left: nodes += self.left.collect_nodes()
            if self.right: nodes += self.right.collect_nodes()
            return nodes
    
        def crossover(self, other):
            if random.random() > CROSSOVER_RATE:
                return
            n1 = random.choice(self.collect_nodes())
            n2 = random.choice(other.collect_nodes())
            (n1.data, n1.left, n1.right, n2.data, n2.left, n2.right) = (
                n2.data, n2.left, n2.right, n1.data, n1.left, n1.right
            )
    
        def mutate(self):
            if random.random() < MUTATION_RATE:
                new_tree = GPTree()
                new_tree.random_subtree(grow=True, max_depth=MAX_DEPTH)
                self.data, self.left, self.right = new_tree.data, new_tree.left, new_tree.right
            else:
                if self.left: self.left.mutate()
                if self.right: self.right.mutate()
    
        #  Postfix & Infix Representation 
        def to_postfix(self):
            if callable(self.data):
                return self.left.to_postfix() + self.right.to_postfix() + [self.node_label()]
            return [str(self.data)]
    
        def to_infix(self):
            if callable(self.data):
                return f"({self.left.to_infix()} {self.node_label()} {self.right.to_infix()})"
            return str(self.data)
    
    #  INITIAL POPULATION 
    def init_population():
        population = []
        per_depth = POP_SIZE // ((MAX_DEPTH - MIN_DEPTH + 1) * 2)
        for depth in range(MIN_DEPTH, MAX_DEPTH + 1):
            for _ in range(per_depth):
                t = GPTree()
                t.random_subtree(grow=True, max_depth=depth)
                population.append(t)
            for _ in range(per_depth):
                t = GPTree()
                t.random_subtree(grow=False, max_depth=depth)
                population.append(t)
        return population
    
    #  DATASET 
    def get_dataset():
        return [
            (-1, -1, -6.33333), (-1, 0, -6), (-1, 1, -5.66667), (-1, 2, -5.33333), (-1, 3, -5),
            (-1, 4, -4.66667), (-1, 5, -4.33333), (0, -1, -4.33333), (0, 0, -4), (0, 1, -3.66667),
            (0, 2, -3.33333), (0, 3, -3), (0, 4, -2.66667), (0, 5, -2.33333), (1, -1, -2.33333),
            (1, 0, -2), (1, 1, -1.66667), (1, 2, -1.33333), (1, 3, -1), (1, 4, -0.666667),
            (1, 5, -0.333333), (2, -1, -0.333333), (2, 0, 0), (2, 1, 0.333333), (2, 2, 0.666667),
            (2, 3, 1), (2, 4, 1.33333), (2, 5, 1.66667), (3, -1, 1.66667), (3, 0, 2), (3, 1, 2.33333),
            (3, 2, 2.66667), (3, 3, 3), (3, 4, 3.33333), (3, 5, 3.66667), (4, -1, 3.66667),
            (4, 0, 4), (4, 1, 4.33333), (4, 2, 4.66667), (4, 3, 5), (4, 4, 5.33333), (4, 5, 5.66667),
            (5, -1, 5.66667), (5, 0, 6), (5, 1, 6.33333), (5, 2, 6.66667), (5, 3, 7), (5, 4, 7.33333),
            (5, 5, 7.66667)
        ]
    
    
    
    #### Fitness Function
    #  FITNESS 
    def fitness(individual, dataset):
        errors = []
        for x, y, result in dataset:
            try:
                pred = individual.evaluate(x, y)
            except:
                pred = 0
            errors.append((pred - result) ** 2)
        return sum(errors) / len(errors)
    
    
    
    #### Selection
    #  TOURNAMENT SELECTION 
    def tournament_selection(population, dataset, k=TOURNAMENT_SIZE):
        competitors = random.sample(population, k)
        competitors.sort(key=lambda ind: fitness(ind, dataset))
        return competitors[0]
    
    
    
    
    #### Evolutionary Loop
    #  MAIN GP LOOP WITH ELITISM & PLOT & EXPRESSIONS 
    DESIRED_PRECISION = 0.001
    def evolve_with_plot():
        random.seed()
        dataset = get_dataset()
        population = init_population()
        best, best_err = None, float('inf')
        best_gen = 0
        fitness_history = []
        for gen in range(1, GENERATIONS + 1):
            scored = [(fitness(ind, dataset), ind) for ind in population]
            scored.sort(key=lambda x: x[0])
            err, champ = scored[0]
            fitness_history.append(err)
            if err < best_err:
                best_err = err
                best = champ.copy()
                best_gen = gen
                print(f"[Gen {gen:03d}] New best MSE = {best_err:.6f}")         
    #### Termination and results
            if best_err <= DESIRED_PRECISION:
                break
            # Elitism
            new_pop = [best.copy() for _ in range(ELITISM_SIZE)]
            # Create new pop until the number of pop is enough
            while len(new_pop) < POP_SIZE:
                p1 = tournament_selection(population, dataset)
                p2 = tournament_selection(population, dataset)
                child = p1.copy()
                child.crossover(p2)
                child.mutate()
                new_pop.append(child)
            population = new_pop
    
        print("\n=== Best Expression Tree ===")
        best.print_tree()
        print("\nPostfix Notation:")
        print(" ".join(best.to_postfix()))
        print("\nInfix Expression:")
        print(best.to_infix())
        print(f"\nFinal MSE: {best_err:.6f} at generation {best_gen}")
    
        # Biểu đồ MSE theo thế hệ
        plt.plot(fitness_history, label="Best MSE per Generation", color='blue')
        plt.xlabel("Generation")
        plt.ylabel("MSE")
        plt.title("Genetic Programming Fitness Evolution")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    if __name__ == "__main__":
        evolve_with_plot()
