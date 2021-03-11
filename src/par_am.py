import random

import numpy as np

from par_ag import AG

class AM(AG):

    epsilon = None # ξ = 0,1·n
    gen_freq = None
    ls_size = None
    better_ls = None

    # ls
    neighbours = []
    ls_access = []
    local_optimization = None

    def __init__(self, f_name, n_rest, k, seed, gen_freq, ls_size, better):
        AG.__init__(self, f_name, n_rest, k, seed, 'ag', 'g', "un")
        self.epsilon = 0.1*self.n_instances
        # soft local search frequenzy in one AM execution
        self.gen_freq = int(gen_freq)
        # how many chromosomes will be optimized
        self.ls_size = int(float(ls_size)*self.chromosomes)
        #self.ls_access = random.sample(range(0, self.n_instances-1), self.ls_size)
        if better == "better":
            self.better_ls = better
            self.local_optimization = self.local_optimizacion_10
        else:
            self.local_optimization = self.local_optimization_normal

    def am(self):
        self.initialize_population()
        self.evaluate_population()
        i = 0

        while self.evaluations < 100000:
            i += 1
            self.select_function()  # selection
            self.cross_f()  # recombination
            self.mutation_function()
            self.evaluate_newgen()
            if i == self.gen_freq: # mirar si poner esto tras replace y cambiar self.new_generation por self.population
                i = 0
                self.local_optimization()
            self.replace_function()  # search worst chromosome and repalce with best from past generation
            self.evaluate_population()
            #print("Actual n of evaluations: " + str(self.evaluations))

        self.update_data(self.get_best_population())

    def local_optimization_normal(self):
        #random.shuffle(self.ls_access)
        #print(len(range(0, self.chromosomes)))
        #print(self.ls_size)
        self.ls_access = random.sample(range(0, self.chromosomes), self.ls_size)
        for i in range(self.ls_size):
            #print(i)
            self.soft_ls(self.ls_access[i])

    def local_optimizacion_10(self):
        top10 = np.array(self.f_newg)
        top10 = top10.argsort()[:self.ls_size]
        for i in range(len(top10)):
            self.soft_ls(top10[i])

    def soft_ls(self, chromosome): # chromosome -> S in LS
        random.shuffle(self.rsi)
        mis = 0
        improve = True
        i = 0
        while (improve == True or mis < self.epsilon) and i < self.n_instances:
            #print(i)
            mejora = self.min_cluster(chromosome, self.rsi[i])
            if mejora == False:
                mis += 1
            i += 1

    def min_cluster(self, chromosome, instance):
        actual_f = self.f_newg[chromosome]
        #actual_cluster = self.new_generation[chromosome][instance]
        improve = False
        for i in range(self.k):
            go_back = 0
            actual_cluster = self.new_generation[chromosome][instance]
            if i != actual_cluster:

                self.new_generation[chromosome][instance] = i
                if self.is_factible(self.new_generation[chromosome])[0] == 1:
                    ci_f = self.evaluate_chromosome(self.new_generation[chromosome])
                    if ci_f < actual_f:
                        actual_f = ci_f
                        improve = True
                    else:

                        go_back = 1
                else:
                    go_back = 1
                # if not a valid solution to evaluate or just not better, return to old value
                if go_back == 1:
                    self.new_generation[chromosome][instance] = actual_cluster

        return improve
