from random import random, randint

class Perceptron:
    def __init__(self, neurons, initial_bias=1, alpha=1, theta=0.2, zero_weights=False):
        self.neurons = neurons
        self.alpha = alpha
        self.theta = theta
        if zero_weights:
            self.weights = self.generate_zero_weights()
        else:
            self.weights = self.generate_random_weights()
        self.bias = initial_bias

    def train(self, training_data, targets, epochs=5):
        trained = False
        errors = 0
        epoch = 0
        for _ in range(epochs):
            if not trained:
                trained = True
                for i, v in enumerate(training_data):
                    y = self.calculate_output(v)
                    if y != targets[i]:
                        errors += 1
                        self.update_weights(v, targets[i])
                        self.update_bias(targets[i])
            if errors is not 0:
                trained = False
                errors = 0
            epoch += 1

            if trained:
                break

        return epoch

    def calculate_output(self, inputs):
        y_in = self.calculate_weighted_inputs(inputs)
        y = self.apply_activation_function(y_in)

        return y

    def run(self, inputs):
        y = self.calculate_output(inputs)

        return y

    def generate_random_weights(self):
        weights = []
        for i in range(self.neurons):
            weights.append(random())

        return weights

    def generate_zero_weights(self):
        weights = []
        for i in range(self.neurons):
            weights.append(0.0)

        return weights

    def calculate_weighted_inputs(self, inputs):
        y_in = 0
        for i in range(len(inputs)):
            y_in = y_in + (inputs[i] * self.weights[i])

        return y_in + self.bias

    def apply_activation_function(self, y_in):
        if y_in > self.theta:
            return 1
        elif -self.theta <= y_in <= self.theta:
            return 0
        elif y_in < self.theta:
            return -1

    def update_weight(self, x, current_weight, target):
        delta_w = self.alpha * x * target

        return current_weight + delta_w

    def update_weights(self, inputs, target):
        for i in range(len(self.weights)):
            x = inputs[i]
            current_weight = self.weights[i]

            self.weights[i] = self.update_weight(x, current_weight, target)

    def update_bias(self, target):
        delta_b = self.alpha * target

        self.bias = self.bias + delta_b

class Chromosome:
    def __init__(self, bias, alpha, theta, epochs, neurons, inputs, targets, mutation_rate):
        self.bias = bias
        self.alpha = alpha
        self.theta = theta
        self.epochs = epochs
        self.inputs = inputs
        self.targets = targets
        self.neurons = neurons
        self.mutation_rate = mutation_rate
        self.fitness = 0
        self.fitness_in_100_scale = 0
        self.epoch_run = 0

        # It's better to set the zero_weights parameter to True
        # in the Chromosome so that the fitness value could be measured
        
        self.perceptron = Perceptron(
            neurons=neurons,
            initial_bias=bias,
            alpha=alpha,
            theta=theta,
            zero_weights=True
        )

    def __repr__(self):

        return f"Chromosome(bias={self.bias},alpha={self.alpha},theta={self.theta},fitness={self.fitness})"

    def calculate_fitness(self):
        self.epoch_run = self.perceptron.train(self.inputs, self.targets, self.epochs)
        fitness = 1 - (self.epoch_run / self.epochs)

        self.fitness = fitness
        self.fitness_in_100_scale = round(fitness, 2) * 100
        return fitness

    def crossover(self, other):
        index = randint(0, 2)

        if index == 0:
            return Chromosome(
                self.bias,
                other.alpha,
                other.theta,
                self.epochs,
                self.neurons,
                self.inputs,
                self.targets,
                self.mutation_rate
            )
        if index == 1:
            return Chromosome(
                other.bias,
                self.alpha,
                other.theta,
                self.epochs,
                self.neurons,
                self.inputs,
                self.targets,
                self.mutation_rate
            )
        if index == 2:
            return Chromosome(
                other.bias,
                other.alpha,
                self.theta,
                self.epochs,
                self.neurons,
                self.inputs,
                self.targets,
                self.mutation_rate
            )

    def copy(self):
        return Chromosome(
            self.bias,
            self.alpha,
            self.theta,
            self.epochs,
            self.neurons,
            self.inputs,
            self.targets,
            self.mutation_rate
        )

    def mutate(self):
        if random() > self.mutation_rate:
            index = randint(0, 2)
            if index == 0:
                self.bias = random()
            if index == 1:
                self.alpha = random()
            if index == 2:
                self.theta = random()

class NeuroEvolution:
    def __init__(self, population_count, neurons, inputs, targets, epochs, mutation_rate, max_iteration):
        self.neurons = neurons
        self.inputs = inputs
        self.targets = targets
        self.epochs = epochs
        self.max_iteration = max_iteration
        self.population_count = population_count
        self.mutation_rate = mutation_rate
        self.generation = 0
        self.best_perceptron = None

    def generate_initial_populations(self):
        population = []

        for _ in range(self.population_count):
            initial_bias = random()
            alpha = random()
            theta = random()

            chromosome = Chromosome(
                bias=initial_bias,
                alpha=alpha,
                theta=theta,
                neurons=self.neurons,
                epochs=self.epochs,
                inputs=self.inputs,
                targets=self.targets,
                mutation_rate=self.mutation_rate
            )

            population.append(chromosome)
        return population

    def run(self):
        self.population = self.generate_initial_populations()
        self.calculate_fitnesses(self.population)

        for _ in range(self.max_iteration):
            parent_candidates = self.selection(self.population)
            next_generation = self.crossover(parent_candidates)
            self.mutation(next_generation)

            self.calculate_fitnesses(next_generation)
            best_chromosome = self.get_best_chromosome(next_generation)

            if (self.best_perceptron is None) or (best_chromosome.fitness > self.best_perceptron.fitness):
                self.best_perceptron = best_chromosome


            self.population = next_generation
            self.generation += 1
            print(f"Generation {self.generation} with best chromosome {best_chromosome}")

        print(f"Iteration completed with best chromosome = {self.best_perceptron}")
        print(f"With max epochs of {self.best_perceptron.epoch_run}")

    def calculate_fitnesses(self, population):
        for i in population:
            i.calculate_fitness()

    def selection(self, population):
        max_parent = self.population_count // 2
        parents = []
        total = 0

        for p in population:
            total += p.fitness_in_100_scale

        for _ in range(max_parent):
            random_pick = randint(0, total)
            current_count = 0
            for p in population:
                current_count += p.fitness_in_100_scale
                if random_pick <= current_count:
                    parents.append(p)
                    break
        
        return parents

    def crossover(self, parents):
        childs = []
        parents_count = len(parents)
        for i in range(self.population_count):
            random_index1 = randint(0, parents_count - 1)
            random_index2 = randint(0, parents_count - 1)

            child = parents[random_index1].crossover(parents[random_index2])

            childs.append(child)

        return childs

    def mutation(self, childs):
        for child in childs:
            child.mutate()

    def get_best_chromosome(self, chromosomes):
        best_val = -1
        best_index = 0
        for i in range(len(chromosomes)):
            if chromosomes[i].fitness > best_val:
                best_val = chromosomes[i].fitness
                best_index = i

        return chromosomes[best_index]

data = [
    [1, 1, 1],

    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],

    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],

    [0, 0, 0]
]

targets = [
    1,

    -1,
    -1,
    -1,

    -1,
    -1,
    -1,

    -1
]


def run_single_perceptron():
    perceptron = Perceptron(
        neurons=3,
        initial_bias=1,
        alpha=1,
        theta=0.2,
        zero_weights=True
    )

    print("Training single perceptron...")
    epoch = perceptron.train(training_data=data,targets=targets,epochs=200)
    print(f"Training done in {epoch} epochs")
    print("Result : ")

    for v in data:
        output = perceptron.run(v)
        print(f"{v[0]} && {v[1]} && {v[2]} = {output}")

def run_neuro_evolution():
    ne = NeuroEvolution(
        population_count=20,
        neurons=3,
        inputs=data,
        targets=targets,
        epochs=100,
        mutation_rate=0.25,
        max_iteration=1000
    )

    ne.run()

# run_single_perceptron()
run_neuro_evolution()
