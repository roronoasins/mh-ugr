import math
import random
import sys
from collections import Counter
#from numba import jit

import matplotlib.pyplot as plt
import copy

from par_class import Par

'''
Esquema de evolucion: se seleccionará una población de padres del mismo tamaño que la población genética

Operador de seleccion: Torneo binario -> elegir aleatoriamente dos individuos de la población y seleccionar el mejor de ellos.
En este caso se aplicarán tantos torneos como individuos existan en la población genética, incluyendo los individuos ganadores en la población de padres.

Esquema de reemplazamiento: la población de hijos sustituye automáticamente a la actual.
Para conservar el elitismo, si la mejor solución de la generación anterior no sobrevive, sustituye directamente la peor solución de la nueva población.

Operador de cruce: Se emplearán dos operadores de cruce para representación real, Uno de ellos será el operador uniforme mientras que el otro será el cruce por segmento fijo
Esto resultará en el desarrollo de dos generacionales (AGG-UN y AGG-SF).

Operador de mutación: operador de mutación uniforme.
Supongamos que hemos determinado que debe mutar un gen en un cromosoma. Basta con generar dos números aleatorios, uno para determinar el gen que muta (en el intervalo {0, …, 𝑛−1})
y otro para determinar el nuevo valor (que debe ser distinto del actual y mantener las restricciones). Coincidirá por tanto con el operador de generación de vecinos de la BL.

El tamaño de la población será de 50 cromosomas. La probabilidad de cruce será 0,7 en el AGG y 1 en el AGE (siempre se cruzan los dos padres).
La probabilidad de mutación (por gen) será de 0,001 en ambos casos. El criterio de parada en las dos versiones del AG consistirá en realizar 100000 evaluaciones de la función objetivo.

'''

class AG(Par):

    chromosomes = 50
    population = [] # P(t)
    f_population = []
    f_newg = []
    new_generation = [] # P'
    crossing_over = 0
    cross_prob = 0.7
    gen_mutations = 0
    mut_prob = 0.001
    type = ''
    cross_type = ''
    cross_f = None
    cross_function = None
    mutation_function = None
    select_type = ''
    select_function = None
    replace_function = None
    solution = []

    # f() evaluations
    evaluations = 0

    '''
    cambiar el init para ver si es aggun o aggsf.
    '''
    def __init__(self, f_name, n_rest, k, seed, alg, st, ct):
        Par.__init__(self, f_name, n_rest, k, seed)
        self.restriction_to_list()
        self._lambda = self.get_lambda_value()
        #self.create_population()
        self.crossing_over = int((self.chromosomes/2)*self.cross_prob)
        self.gen_mutations = int(self.chromosomes*self.n_instances*self.mut_prob)
        self.type = alg
        if st == 'g':
            self.cross_f = self.generational_cross
            self.select_function = self.generational_select
            self.replace_function = self.replace_genetical_gen
            self.mutation_function = self.generational_mutation
        if st == 'e':
            #self.gen_mutations = int(2 * self.n_instances * self.mut_prob)
            self.cross_f = self.seasonal_cross
            self.select_function = self.steady_gate_selection
            self.replace_function = self.replace_generation_steady_gate
            self.mutation_function = self.steady_gate_mutation

        if ct == 'un':
            self.cross_function = self.get_uniform_crossing_child
        elif ct == 'sf':
            self.cross_function = self.get_fixed_segment_child


    def ag(self):
        self.initialize_population()
        self.evaluate_population()

        while self.evaluations < 100000:
            self.select_function() # selection
            self.cross_f() # recombination
            self.mutation_function()
            self.evaluate_newgen()
            self.replace_function() # search worst chromosome and repalce with best from past generation
            self.evaluate_population()
            #print("Actual n of evaluations: " + str(self.evaluations))

        self.update_data(self.get_best_population())

    def initialize_population(self):
        for i in range(self.chromosomes):
            self.population.append(self.get_random_chromosome().copy())

    def evaluate_population(self):
        if len(self.f_population) > 0:
            self.f_population.clear()
        for i in range(len(self.population)):
            self.f_population.append(self.evaluate_chromosome(self.population[i]))

    def evaluate_newgen(self):
        if len(self.f_newg) > 0:
            self.f_newg.clear()
        for i in range(len(self.new_generation)):
            self.f_newg.append(self.evaluate_chromosome(self.new_generation[i]))

    def get_random_chromosome(self):
        chromosome = []
        #for i in range(self.n_instances):
            #chromosome.append(random.randint(0,self.k-1))
        #chromosome = random.sample(range(0, self.n_))
        chromosome = [random.randint(0,self.k-1) for iter in range(self.n_instances)]
        while self.validate_actual_chromosome(chromosome) == 0:
            chromosome.clear()
            for i in range(self.n_instances):
                chromosome.append(random.randint(0,self.k-1))
        return chromosome

    def validate_actual_chromosome(self, chromosome):
        #for j in range(self.k):
        #    if chromosome.count(j) == 0: # if empty cluster
        count = Counter(chromosome)
        for i in range(self.k):
            if count[i] == 0:
                return 0
        return 1

    def generational_select(self):
        self.new_generation.clear()
        for i in range(self.chromosomes):
            #c1 = .randint(0, self.chromosomes-1)
            #c2 = random.randint(0, self.chromosomes-1)
            parents = random.sample(range(0, self.chromosomes), 2)
            self.new_generation.append(self.bin_tournament(parents))

    def steady_gate_selection(self):
        # copiar population pero remplazando los dos peores por los dos offsprings
        self.new_generation.clear()

        for i in range(2):
            parents = random.sample(range(0, self.chromosomes), 2)
            self.new_generation.append(self.bin_tournament(parents).copy())

    def bin_tournament(self, parents):
        if self.f_population[parents[0]] < self.f_population[parents[1]]:
            return self.population[parents[0]]
        else:
            return self.population[parents[1]]

    '''
    generamos 𝑛/2 números aleatorios distintos en el rango {0,…,𝑛−1}.
    Seleccionamos uno de los padres y copiamos en la descendencia los genes cuyo índice coincide con los números generados.
    Los genes que quedan por asignar en la descendencia se copian del segundo padre

    generar n/2 aleatorios, en un for in range chromosome.len if i == aleatorio se añade del padre 1, si no del padre 2

    para generar 2 hijos se ejecuta dos veces de los mismos padres
    '''

    def generational_cross(self):
        parent1 = 0
        parent2 = 1
        #print(int(self.crossing_over))
        for i in range(int(self.crossing_over)):
            #child1 = self.cross_function(parent1, parent2)
            #child2 = self.cross_function(parent1, parent2)
            # children replace new_generation
            self.new_generation[parent1] = self.cross_function(parent1, parent2).copy()
            self.new_generation[parent2] = self.cross_function(parent1, parent2).copy()
            # new chromosomes cross
            parent1 += 2
            parent2 += 2

    # only 2 chromosomes in new_generation to cross and mutate(also replaced if better than 2 worst from P(t)
    def seasonal_cross(self):
        # crossing_prob = 1
        parent1 = 0
        parent2 = 1
        child1 = self.cross_function(parent1, parent2)
        child2 = self.cross_function(parent1, parent2)
        self.new_generation[parent1] = child1.copy()
        self.new_generation[parent2] = child2.copy()


    def get_uniform_crossing_child(self, p1, p2):
        #parent_1 = []
        child = [0] * self.n_instances
        parent_1 = random.sample(range(0, self.n_instances), int(self.n_instances/2))
        for j in range(self.n_instances):
            if j in parent_1:
                child[j] = self.new_generation[p1][j]
            else:
                child[j] = self.new_generation[p2][j]

        factible = self.is_factible(child)
        if factible[0] == 0:
            #print("no factible")
            child = self.repair(child, factible[1])

        return child

    def modPow2(self, n, p2):
        return n & (p2 - 1)

    def isPow2(self, n):
        return ((n - 1) & n) == 0
    # https://lustforge.com/2016/05/08/modulo-operator-performance-impact/ http://blog.teamleadnet.com/2012/07/faster-division-and-modulo-operation.html
    def modFast(self, n, b):
        if (self.isPow2(b)):
            self.modPow2(n, b)
        else:
            return n % b

    def get_fixed_segment_child(self, p1, p2):
        # segment copied to p1 by default(random selection reusing)
        child = [0] * self.n_instances

        segment = random.sample(range(0, self.n_instances), 2) # [start, len]
        start = segment[0]
        #limit = (segment[0] + segment[1]) % self.n_instances
        limit = self.modFast(segment[0]+segment[1], self.n_instances)
        # n_gens - segment_len = remaining gens (inherited by child)
        parent_1 = random.sample(range(0, self.n_instances-segment[1]), int((self.n_instances-segment[1])/2))
        for i in range(len(parent_1)):
            #parent_1[i] = (parent_1[i] + start) % self.n_instances
            parent_1[i] = self.modFast(parent_1[i] + limit, self.n_instances)
        for j in range(self.n_instances):
            # {𝑟,((𝑟+𝑣)𝑚𝑜𝑑𝑛)−1}
            if start < j < limit:
                child[j] = self.new_generation[p1][j]   # best parent -> fixed segment? test
            elif j in parent_1:
                child[j] = self.new_generation[p1][j]
            else:
                child[j] = self.new_generation[p2][j]

        factible = self.is_factible(child)
        if factible[0] == 0:
            #print("no factible")
            child = self.repair(child, factible[1])

        return child

    def is_factible(self, chromosome):
        #for j in range(self.k):
        #    if chromosome.count(j) == 0: # if empty cluster
        count = Counter(chromosome)
        for i in range(self.k):
            if count[i] == 0:
                return (0, i)
        return (1, 0)

    def repair(self, chromosome, empty_k):
        #index = random.randint(0, self.n_instances-1)
        chromosome[random.randint(0, self.n_instances-1)] = empty_k

        factible = self.is_factible(chromosome)
        while factible[0] == 0:
            #index = random.randint(0, self.n_instances - 1)
            print(factible[1])
            chromosome[random.randint(0, self.n_instances-1)] = factible[1] # empty_k
            factible = self.is_factible(chromosome)
        return chromosome


    def generational_mutation(self):
        for i in range(int(self.gen_mutations)):
            chromosome = random.randint(0, len(self.new_generation)-1)
            gen = random.randint(0, self.n_instances-1)
            new_gen = random.randint(0, self.k-1)
            valid = 0
            while self.new_generation[chromosome][gen] == new_gen:
                new_gen = random.randint(0, self.k - 1)
            self.new_generation[chromosome][gen] = new_gen

            factible = self.is_factible(self.new_generation[chromosome])
            if factible[0] == 0:
                #print("no factible mut")
                self.new_generation[chromosome] = self.repair(self.new_generation[chromosome], factible[1]).copy()

            #while valid == 0:
             #   child = self.repair(child, factible[1])
              #  valid = self.validate_actual_chromosome(self.new_generation[chromosome])

    def steady_gate_mutation(self):
        for i in range(len(self.new_generation)):
            for j in range(self.n_instances):
                if random.random() <= self.mut_prob:
                    new_gen = random.randint(0, self.k - 1)

                    while self.new_generation[i][j] == new_gen:
                        new_gen = random.randint(0, self.k - 1)
                    self.new_generation[i][j] = new_gen

                    factible = self.is_factible(self.new_generation[i])
                    if factible[0] == 0:
                        self.new_generation[i] = self.repair(self.new_generation[i], factible[1])


    def replace_genetical_gen(self): # , best_past_gen
        #if self.type == 'aggun':
        #if best_past_gen not in self.new_generation:
        #best_past_gen = self.get_best_population()
        self.new_generation[self.get_worst_chromosome()] = self.get_best_population().copy()
        #for i in range(self.chromosomes):
         #  self.population[i] = self.new_generation[i].copy()
        self.population = copy.deepcopy(self.new_generation)
        self.new_generation.clear()
        self.c_array.clear()

    def replace_generation_steady_gate(self):
        worst_new_gen = self.get_worst_chromosome()
        #best_new_gen = self.modFast(worst_new_gen+1,2) # (worst_new_gen + 1) mod 2
        best_new_gen = (worst_new_gen+1)%2
        twoworst_population = self.get_2worst()
        lowest_worst = twoworst_population[1]
        highest_worst = twoworst_population[0]

        #two_best = self.get_2best([(self.f_newg[worst_new_gen], worst_new_gen), (self.f_newg[best_new_gen], best_new_gen), twoworst_population[0], twoworst_population[1]])
        if self.f_newg[best_new_gen] < lowest_worst[0]:
            self.replace_chromosome(best_new_gen, lowest_worst[1])
            if self.f_newg[worst_new_gen] < highest_worst[0]:
                self.replace_chromosome(worst_new_gen, highest_worst[1])
        elif self.f_newg[best_new_gen] < highest_worst[0]:
            self.replace_chromosome(best_new_gen, highest_worst[1])

    def replace_chromosome(self, chromosome, replaced):
        self.population[replaced] = self.new_generation[chromosome].copy()
        self.f_population[replaced] = self.f_newg[chromosome]

    def get_2worst(self):

        worst1 = (self.f_population[0], 0)
        #worst2 = tuple(worst1)
        worst2 = tuple(worst1)

        for i in range(0, self.chromosomes):
            actual_chromosome = (self.f_population[i], i)
            if actual_chromosome[0] > worst1[0]:
                worst2 = tuple(worst1)
                worst1 = tuple(actual_chromosome)
            elif actual_chromosome[0] > worst2[0] and actual_chromosome[0] != worst1[0]:
                worst2 = tuple(actual_chromosome)
        #return worst1, worst2
        return worst1, worst2

    def get_best_population(self):
        #self.evaluations += 1
        #best_chromosome = (self.evaluate_chromosome(self.population[0]), self.population[0])
        best_chromosome = (self.f_population[0], self.population[0])

        for i in range(self.chromosomes):
            #self.evaluations += 1
            #actual_chromosome = (self.evaluate_chromosome(self.population[i]), self.population[i])
            actual_chromosome = (self.f_population[i], self.population[i])
            if actual_chromosome[0] < best_chromosome[0]:
                #print(actual_chromosome[0])
                best_chromosome = actual_chromosome
            #best_chromosome = self.return_better_chromo(actual_chromosome, best_chromosome)
        #print("Best past gen: "+str(best_chromosome[1]))
        return best_chromosome[1]

    def get_worst_chromosome(self):
        worst_chromosome = (self.f_newg[0], 0)

        for i in range(len(self.new_generation)-1):
            actual_chromosome = (self.f_newg[i], i)
            if actual_chromosome[0] > worst_chromosome[0]:
                worst_chromosome = tuple(actual_chromosome)

        return worst_chromosome[1]

    def evaluate_chromosome(self, chromosome):
        self.get_chromosome_centroids(chromosome)
        self.evaluations += 1

        f = self.f(chromosome, self.get_distance_to_centroids())
        #print(f)
        print(self.evaluations)
        return f

    '''
    A chromosome is better than another if has less f(). If f value is equal, ¿?
    '''
    def return_better_chromo(self, c1, c2):
        if c1[0] < c2[0]:
            return c1
        else:
            return c2

    def f(self, chromosome, distances):
        return self.get_general_deviation_ag(chromosome, distances) + (self.get_infeasibility(chromosome) * self._lambda)

    def get_general_deviation(self, chromosome):
        mean_distance = [0] * self.k

        for i in range(self.n_instances):
            mean_distance[chromosome[i]] += self.k_distances[i][chromosome[i]]

        general_deviation = 0
        for k in range(self.k):
            mean_distance[k] /= chromosome.count(k)
            general_deviation += mean_distance[k]
        general_deviation /= self.k
        return general_deviation

    def get_general_deviation_ag(self, chromosome, distances):
        mean_distance = [0] * self.k

        for i in range(self.n_instances):
            mean_distance[chromosome[i]] += distances[i][chromosome[i]]

        general_deviation = 0
        for k in range(self.k):
            mean_distance[k] /= chromosome.count(k)
            general_deviation += mean_distance[k]
        general_deviation /= self.k
        return general_deviation

    def get_chromosome_centroids(self, chromosome):
        # update c_array
        if len(self.c_array) > 0:
            self.c_array.clear()
        self.create_carray()

        for b in range(len(chromosome)):
            self.c_array[chromosome[b]].append(b)

        # update centroids data
        if len(self.centroids) > 0:
            self.centroids.clear()
        for c in range(self.k):
            self.centroids.append([0] * self.features)
        for i in range(self.k):
            av = [0] * self.features
            for j in range(len(self.c_array[i])):
                for k in range(self.features):
                    av[k] = av[k] + self.data[k][self.c_array[i][j]]
                    for k in range(self.features):
                        self.centroids[i][k] = av[k] / len(self.c_array[i])
        #return centroids

    def get_distance_to_centroids(self): # (self, centroids)
        distances = []
        for i in range(self.n_instances):
            distances.append([])
            for j in range(self.k):
                dist = 0
                for k in range(self.features):
                    dist += (self.data[k][i] - self.centroids[j][k]) ** 2
                distances[i].append(math.sqrt(dist))
        return distances

    def get_infeasibility(self, chromosome):
        inf = 0
        for i in range(len(self.restrictions_list)):
            c1 = chromosome[self.restrictions_list[i][0]]
            c2 = chromosome[self.restrictions_list[i][1]]
            rest = self.restrictions_list[i][2]
            if rest == 1 and c1 != c2:
                inf += 1
            elif rest == -1 and c1 == c2:
                inf += 1
        return inf

    def update_data(self, chromosome):
        self.solution = chromosome.copy()
        self.evaluate_chromosome(self.solution)

    def solution_inf(self):
        return self.get_infeasibility(self.solution)

    def solution_deviation(self):
        for i in range(self.n_instances):
            self.k_distances.append([])
            for j in range(self.k):
                self.k_distances[i].append([0] * self.k)
        self.update_distances()
        return self.get_general_deviation(self.solution)

    def solution_f(self):
        return self.f(self.solution, self.get_distance_to_centroids())

    def print_solution(self):
        print(self.solution)
