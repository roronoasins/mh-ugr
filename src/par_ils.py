from par_ls import LS
from par_sals import SA

import random
import math

class ILS(SA):
    iterations = 10
    max_evaluations = 10000
    best_f_ils = 999999
    best_s = []

    def __init__(self, f_name, n_rest, k, seed, search):
        if search == 'ls':
            LS.__init__(self, f_name, n_rest, k, seed)
            self.search = self.local_search
        elif search == 'sa':
            SA.__init__(self, f_name, n_rest, k, seed)
            self.search = self.sa
            self.max_success /= 10
            self.max_neighbours /= 10
            print(self.max_neighbours)
            print(self.max_success)
            self.M = self.max_evaluations/self.max_neighbours

        self.mutation_value = 0.1*self.n_instances


    def ils(self):
        self.get_initial_solution()
        s_data = self.search()
        self.best_f_ils = s_data[0]
        self.best_s = s_data[1].copy()

        i = 1
        print("Iteration 1 done")
        print("S: "+str(s_data[0]))
        print("Best: "+str(self.best_f_ils))
        while i < self.iterations:
            self.modify(s_data[0])
            new_result = self.search()
            s_data = self.accept(s_data, new_result)
            self.update(s_data)
            i += 1
            print("Iteration "+str(i)+" done")
            print("S: "+str(s_data[0]))
            print("Best: "+str(self.best_f_ils))

        self.S = self.best_s.copy()
        self.update_data()


    def accept(self, s_struct, new_struct):
        if new_struct[0] < s_struct[0]:
            return new_struct
        else:
            return s_struct

    def update(self, s_struct):
        if s_struct[0] < self.best_f_ils:
            self.best_f_ils = s_struct[0]
            self.best_s = s_struct[1].copy()

    def modify(self, history):
        # select the actual best solution
        if(history > self.best_f_ils):
            self.S = self.best_s.copy()
            self.update_data()
        # segment start
        r = random.randint(0, self.n_instances-1)
        f = (r+self.mutation_value)%self.n_instances - 1

        for i in range(self.n_instances):
            if i > r or i < f:
                self.change_cluster(i, random.randint(0, self.k-1))

    def change_cluster(self, i, k):
        old_k = self.S[i]
        self.S[i] = k
        while self.validate_actual_solution() == 0:
            k = random.randint(0, self.k-1)
            old_k = self.S[i]
            self.S[i] = k

    def update_data(self):
        self.update_carray()
        self.update_centroids()
        self.update_distances()

    def local_search(self):
        self.neighbours_generation()
        random_access = list(range(0,len(self.neighbours)))
        random.shuffle(random_access)

        actual_f = self.f()
        iteration = 0
        end = False
        while not end and iteration < self.max_evaluations:
            # each evaluation, flag restarted
            end = True
            for i in range(len(self.neighbours)):
                go_back = 0
                actual_index = self.neighbours[random_access[i]][0]
                actual_new_cluster = self.neighbours[random_access[i]][1]
                actual_cluster = self.S[actual_index]
                self.S[actual_index] = actual_new_cluster # (i0,i1) -> (index, cluster)

                if self.validate_actual_solution() == 1:
                    self.update_changed_centroids(actual_cluster, actual_new_cluster)
                    neighbor_f = self.f()
                    iteration += 1
                    if neighbor_f < actual_f:
                        # new solution to expand; new environment creation
                        actual_f = neighbor_f
                        self.neighbours_generation()
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
                    self.S[actual_index] = actual_cluster
                    if return_anterior_state == 1:
                        self.update_changed_centroids(actual_new_cluster, actual_cluster)
                        return_anterior_state = 0
                        #self.update_carray()

        print('\nEvaluations: '+str(iteration))
        # bmb return to get best solution from all 10 iterations
        return self.f(), self.S.copy()

    def sa(self):
        #self.get_initial_solution()
        self.update_carray()
        self.To = self.get_To()
        print("Initial temperature: " + str(self.To))
        T = self.To
        best_s = self.S.copy()
        self.best_f = self.f()

        continue_cooling = False
        stucked = False
        evaluations = 0
        while T > self.Tf and stucked == False and evaluations < self.max_evaluations:
            success = 0
            i = 0
            f_inherited = False
            while i < self.max_neighbours and continue_cooling == False and evaluations < self.max_evaluations:
                #print(evaluations)
                aux_s = self.S.copy()
                f_s = self.f()
                evaluations += 1
                # create method change_cluster(get_neigh)
                old_data = self.get_neighbour()

                old_cluster = old_data[1]
                new_cluster = self.S[old_data[0]]

                self.update_changed_centroids(old_cluster, new_cluster)
                self.c_array[old_cluster].remove(old_data[0])
                self.c_array[new_cluster].append(old_data[0])
                f_candidate_s = self.f()
                evaluations += 1
                inc_f = f_candidate_s - f_s
                a = -inc_f/(T)
                #print(math.exp(a))
                if inc_f < 0 or random.random() <= math.exp(a):
                    f_s = f_candidate_s
                    if f_s < self.best_f:
                        self.best_f = f_s
                        best_s = self.S.copy()
                        success += 1
                # return past state, create method
                else:
                    self.S = aux_s.copy()
                    self.S[old_data[0]] = old_cluster
                    self.c_array[old_cluster].append(old_data[0])
                    self.c_array[new_cluster].remove(old_data[0])
                    self.update_changed_centroids(new_cluster, old_cluster)

                i+=1
                if success == self.max_success:
                    print("reset")
                    continue_cooling = True

                #print("Neigh: " + str(i))

            continue_cooling = False
            if success == 0:
                print("f")
                stucked = True
            T = self.get_next_temperature_cauchy(T)
            #T = self.get_next_temperature_lineal(T)
            print("T:"+str(T))
            print("Tf:"+str(self.Tf))

        #update data estructures with best_S
        self.S = best_s.copy()
        self.update_carray()
        self.update_centroids()
        self.update_distances()
        return self.best_f, best_s.copy()
