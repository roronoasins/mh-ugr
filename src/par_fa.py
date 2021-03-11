from par_ag import AG
import random
import math
from collections import Counter

'''
    Representacion: pense en hacer movimiento de centroides pero a la hora de recrear el vector de asignaciones que representa la solucion,
    el asignar mirando el cluter mas cercano acada elemento no es valido ya que pierdes el valor de tu f. por eso pense en hacer movimiento de
    clusters asignados, de forma que al hacer modulos se pareceria al cruce(por segmento?) y podria funcionar.

    ToDo:
    - Mejoras: cambiar el parametro de aleatoriedad en el movimiento para que funcione como el esquema de enfriamiento del SA
'''

#@jitclass(spec)
class FA(AG):
    n_fireflies = 50
    light_absorption_coefficient = 0
    max_generation = 50
    max_evaluations = 100000
    light_intensity = []
    f_ff = []
    fireflies = []
    real_values_ff = []
    ff_centroids = [] # actual ff centroids(updated with each move)
    ff_k_distances = []
    #ff_distances = [] # each i(firefly) has all distances towards each ff
    freq = 10
    neighbours = []

    best_f = 999999
    best_ff = []

    def __init__(self, f_name, n_rest, k, seed, ls_size, better):
        AG.__init__(self, f_name, n_rest, k, seed, '', '', '')
        #self.create_inital_fireflies()
        self.k_distances = [0] * self.n_instances # fill with zeros
        self.initialize_fireflies()
        self.create_ff_centroids()
        self.get_all_ff_centroids()
        self.create_ff_kdistances()
        self.evaluate_fireflies()
        self.freq = 1
        # n_fireflies+1 -> best_firefly found
        self.fireflies.append([])
        self.light_intensity.append(999999)

        self.ls_size = round(float(ls_size)*self.n_fireflies)
        if better == "better":
            self.better_ls = better
            self.local_optimization = self.local_optimization_better
        else:
            self.local_optimization = self.local_optimization_normal

    '''
    Objective function f(x),x= (x_1, ..., x_d)T
    Generate initial population of ﬁreﬂies xi(i= 1,2, ..., n)
    Light intensity I_i at x_i is determined by f(x_i)
    Deﬁne light absorption coeﬃcient γ
    while (t <MaxGeneration)
        for i= 1 : n all n ﬁreﬂies
            for j= 1 : i all n ﬁreﬂies
                if (Ij> Ii), Move ﬁreﬂy i towards j in d-dimension; end if
                Attractiveness varies with distance r via exp[−γr]
                Evaluate new solutions and update light intensity
            end for j
        end for i
        Rank the ﬁreﬂies and ﬁnd the current best
    end while
    '''
    def fa(self):
        evaluations = 0
        do = 0
        while self.evaluations < self.max_evaluations:
            do += 1
            for i in range(self.n_fireflies):
                brighter = True
                for j in range(self.n_fireflies):
                    if self.light_intensity[j] < self.light_intensity[i]:
                        # Move ﬁreﬂy i towards j
                        self.move_to(i, j)
                        brighter = False
                if brighter == True:
                    self.move_randomly(i)
                print(self.light_intensity[self.best_firefly()])
            if do == self.freq:
                do = 0
                print("soft local search")
                self.local_optimization()
                print("end local search")

    def create_ff_centroids(self):
        for i in range(self.n_fireflies):
            self.ff_centroids.append([])

    def get_all_ff_centroids(self):
        for i in range(self.n_fireflies):
            self.get_chromosome_centroids(self.fireflies[i])
            self.ff_centroids[i] = self.centroids.copy()

    def create_ff_kdistances(self):
        for n in range(self.n_fireflies):
            self.ff_k_distances.append([])
            for i in range(self.n_instances):
                self.ff_k_distances[n].append([0]*self.k)
                for j in range(self.k):
                    dist = 0
                    for k in range(self.features):
                        dist = dist + (self.data[k][i]-self.ff_centroids[n][j][k]) **2
                    self.ff_k_distances[n][i][j] = math.sqrt(dist)

    def evaluate_firefly(self, ff):
        self.evaluations += 1
        f = self.f(ff)
        #print(f)
        print(self.evaluations)
        return f

    def evaluate_fireflies(self):
        if len(self.light_intensity) > 0:
            self.light_intensity.clear()
        for i in range(self.n_fireflies):
            self.light_intensity.append(self.evaluate_firefly(i))

    # if at least one firefly is brighter than i, false is returned. Otherwise, true would be.
    def check_brighter(self, i):
        for j in range(self.n_fireflies):
            if j != i and self.fireflies[j] < self.fireflies[i]:
                return False
        return True

    def f(self, ff):
        return self.get_general_deviation(ff) + (self.get_infeasibility(ff) * self._lambda)

    def get_general_deviation(self, ff):
        mean_distance = [0] * self.k
        for i in range(self.n_instances):
            mean_distance[self.fireflies[ff][i]] += self.ff_k_distances[ff][i][self.fireflies[ff][i]]

        general_deviation = 0
        for k in range(self.k):
            mean_distance[k] /= self.fireflies[ff].count(k)
            general_deviation += mean_distance[k]
        general_deviation /= self.k
        return general_deviation

    def get_infeasibility(self, ff):
        inf = 0
        for i in range(len(self.restrictions_list)):
            c1 = self.fireflies[ff][self.restrictions_list[i][0]]
            c2 = self.fireflies[ff][self.restrictions_list[i][1]]
            rest = self.restrictions_list[i][2]
            if rest == 1 and c1 != c2:
                inf += 1
            elif rest == -1 and c1 == c2:
                inf += 1
        return inf

    def initialize_fireflies(self):
        for i in range(self.n_fireflies):
            self.fireflies.append(self.get_random_chromosome().copy())
            self.real_values_ff.append(self.fireflies[i].copy())

    def fireflies_distance(self, i, j):
        d = 0
        for k in range(self.n_instances):
            d += (self.fireflies[j][k] - self.fireflies[i][k]) **2
        return math.sqrt(d)

    def attractiveness_gaussian_form(self, ff_i, ff_source):
        return self.light_intensity[ff_source] * math.exp(-self.fireflies_distance(ff_i, ff_source)**2)

    def attractiveness_simplified(self, ff_i, ff_source):
        return 1/(1+self.fireflies_distance(ff_i, ff_source))

    def move_to(self, i, j):
        aux = self.fireflies[i].copy()
        for n in range(self.n_instances):
            self.fireflies[i][n] += round(self.attractiveness_simplified(i,j)*self.diff_ff(j, i)+random.randint(0, 1)-0.5)
            self.fireflies[i][n] %= self.k
            #self.real_values_ff[i][n] += self.attractiveness_simplified(i,j)*self.diff_ff(j, i)+random.randint(0, 1)-0.5
            #frac, whole = math.modf(self.real_values_ff[i][n])
            #self.fireflies[i][n] = (whole + self.fireflies[i][n]) % self.k
            #self.real_values_ff[i][n] = (whole%self.k)+frac
        if self.validate_actual_ff(i) == 1:
            self.update_ff_carray(i)
            self.update_ff_centroids(i)
            self.update_ff_kdistances(i)
            self.light_intensity[i] = self.evaluate_firefly(i)
            if self.light_intensity[i] < self.light_intensity[self.n_fireflies]:
                self.light_intensity[self.n_fireflies] = self.light_intensity[i]
                self.fireflies[self.n_fireflies] = self.fireflies[i].copy()
        else:
            self.fireflies[i] = aux.copy()

    def move_randomly(self, i):
        aux = self.fireflies[i].copy()
        for n in range(self.n_instances):
            self.fireflies[i][n] += round(random.randint(0, 1)-0.5)
            self.fireflies[i][n] %= self.k

        if self.validate_actual_ff(i) == 1:
            self.update_ff_carray(i)
            self.update_ff_centroids(i)
            self.update_ff_kdistances(i)
            self.light_intensity[i] = self.evaluate_firefly(i)
            if self.light_intensity[i] < self.best_f:
                self.best_f = self.light_intensity[i]
                self.best_ff = self.fireflies[i].copy()
        else:
            self.fireflies[i] = aux.copy()

    def diff_ff(self, ff_i, ff_j):
        diff = 0
        for n in range(self.n_instances):
            diff += self.fireflies[ff_i][n]-self.fireflies[ff_j][n]
        return diff

    def best_firefly(self):
        return min(range(len(self.light_intensity)), key=self.light_intensity.__getitem__)

    def update_ff_kdistances(self, ff):
            for i in range(self.n_instances):
                for j in range(self.k):
                    dist = 0
                    for k in range(self.features):
                        dist = dist + (self.data[k][i]-self.ff_centroids[ff][j][k]) **2
                    self.ff_k_distances[ff][i][j] = math.sqrt(dist)

    def update_ff_centroids(self, ff):
        for i in range(self.k):
            av = [0] * self.features
            for j in range(len(self.c_array[i])):
                for k in range(self.features):
                    av[k] = av[k] + self.data[k][self.c_array[i][j]]
                for k in range(self.features):
                    self.ff_centroids[ff][i][k] = av[k] / len(self.c_array[i])

    def update_ff_carray(self, ff):
        # update c_array
        if len(self.c_array) > 0:
            self.c_array.clear()
        self.create_carray()

        for b in range(self.n_instances):
            self.c_array[self.fireflies[ff][b]].append(b)


    def local_optimization_normal(self):
        ls_access = random.sample(range(0, self.n_fireflies), self.ls_size)
        for i in range(self.ls_size):
            self.local_search(ls_access[i])

    def local_optimization_better(self):
        index = sorted(range(self.n_fireflies), key = lambda sub: self.light_intensity[sub])[:self.ls_size]
        for i in range(len(index)):
            self.local_search(index[i])

    def soft_ls(self, ff): # chromosome -> S in LS
        random.shuffle(self.rsi)
        mis = 0
        improve = True
        i = 0
        max_miss = 0.1*self.n_instances
        while mis < max_miss and i < self.n_instances:
            #print(i)
            mejora = self.min_cluster(ff, i)
            if mejora == False:
                mis += 1
            i += 1

    def min_cluster(self, ff, instance):
        actual_f = self.light_intensity[ff]
        #actual_cluster = self.new_generation[chromosome][instance]
        improve = False

        actual_cluster = self.fireflies[ff][instance]
        for i in range(self.k):
            go_back = 0

            if i != actual_cluster:

                self.fireflies[ff][instance] = i
                if self.is_factible(self.fireflies[ff])[0] == 1:
                    # cambiar los update por solo los cambios reales, no sobre todo el vector de asignaciones de cada ff
                    self.update_ff_carray(ff)
                    self.update_ff_centroids(ff)
                    self.update_ff_kdistances(ff)
                    ci_f = self.evaluate_firefly(ff)
                    if ci_f < actual_f:
                        actual_f = ci_f
                        self.light_intensity[ff] = actual_f
                        improve = True
                    else:

                        go_back = 1
                else:
                    go_back = 1
                # if not a valid solution to evaluate or just not better, return to old value
                if go_back == 1:
                    self.fireflies[ff][instance] = actual_cluster
                    self.update_ff_carray(ff)
                    self.update_ff_centroids(ff)
                    self.update_ff_kdistances(ff)

        return improve

    def neighbours_generation(self, ff):
        if len(self.neighbours) > 0:
            self.neighbours.clear()
        for i in range(self.n_instances):
            for j in range(self.k):
                if j != self.fireflies[ff][i]:
                    self.neighbours.append((i, j))  # pair(index, value)

    def local_search(self, ff):
        self.neighbours_generation(ff)
        random_access = list(range(0,len(self.neighbours)))
        random.shuffle(random_access)

        actual_f = self.light_intensity[ff]
        iteration = 0
        end = False
        while not end and iteration < 200:
            # each evaluation, flag restarted
            end = True
            return_anterior_state = 0
            for i in range(len(self.neighbours)):
                go_back = 0
                actual_index = self.neighbours[random_access[i]][0]
                actual_new_cluster = self.neighbours[random_access[i]][1]
                actual_cluster = self.fireflies[ff][actual_index]

                self.fireflies[ff][actual_index] = actual_new_cluster # (i0,i1) -> (index, cluster)

                if self.validate_actual_ff(ff) == 1:
                    self.update_changed_centroids(ff, actual_cluster, actual_new_cluster)
                    neighbor_f = self.evaluate_firefly(ff)
                    iteration += 1
                    if neighbor_f < self.light_intensity[ff]:
                        # new solution to expand; new environment creation
                        self.light_intensity[ff] = neighbor_f
                        self.neighbours_generation(ff)
                        random.shuffle(random_access)
                        end = False
                        break # exit for and iterate within new environment
                    else:
                        go_back = 1
                        return_anterior_state = 1
                else:
                    go_back = 1
                # if not a valid solution to evaluate or just not better, return to old value
                if go_back == 1:
                    self.fireflies[ff][actual_index] = actual_cluster
                    if return_anterior_state == 1:
                        self.update_changed_centroids(ff, actual_new_cluster, actual_cluster)
                        return_anterior_state = 0

    def update_changed_centroids(self, ff, old_cluster, new_cluster):
            av_old = [0] * self.features
            av_new = [0] * self.features

            for i in range(self.n_instances):
                if self.fireflies[ff][i] == old_cluster or self.fireflies[ff][i] == new_cluster :
                    for k in range(self.features):
                        if self.fireflies[ff][i] == old_cluster:
                            av_old[k] = av_old[k] + self.data[k][i]
                        elif self.fireflies[ff][i] == new_cluster:
                            av_new[k] = av_new[k] + self.data[k][i]
            for k in range(self.features):
                self.ff_centroids[ff][old_cluster][k] = av_old[k] / self.fireflies[ff].count(old_cluster)
                self.ff_centroids[ff][new_cluster][k] = av_new[k] / self.fireflies[ff].count(new_cluster)

            # update distances(only related clusters)
            for i in range(self.n_instances):
                for j in range(self.k):
                    if j == old_cluster or j == new_cluster:
                        dist = 0
                        for k in range(self.features):
                            dist = dist + (self.data[k][i]-self.ff_centroids[ff][j][k]) **2
                        self.ff_k_distances[ff][i][j] = math.sqrt(dist)
            self.update_ff_carray(ff)

    def update_data(self, ff):
        self.centroids = self.ff_centroids[ff].copy()
        self.k_distances = self.ff_k_distances[ff].copy()
        self.update_ff_carray(ff)
        self.solution = self.fireflies[ff]

    def solution_inf(self):
        return self.get_infeasibility(self.best_firefly())

    def solution_deviation(self):
        ff = self.best_firefly()
        self.centroids = self.ff_centroids[ff].copy()
        self.update_ff_carray(self.best_firefly())
        self.k_distances = self.ff_k_distances[ff].copy()
        return self.get_general_deviation(ff)

    def solution_f(self):
        return self.light_intensity[self.best_firefly()]

    def print_solution(self):
        print(self.fireflies[self.best_firefly()])

    def validate_actual_ff(self, ff):
        count = Counter(self.fireflies[ff])
        for i in range(self.k):
            if count[i] == 0:
                return 0
        return 1
