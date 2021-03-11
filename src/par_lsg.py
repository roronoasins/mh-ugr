import math
import random
import sys
import matplotlib.pyplot as plt
from collections import Counter

from par_greedy import Greedy

class LSG(Greedy):

    ls_solution = []
    neighbours = []
    _lambda = 0
    max_evaluations = 100000

    def __init__(self, f_name, n_rest, k, seed):
        Greedy.__init__(self, f_name, n_rest, k, seed)
        self.generate_centroids()
        self.create_kdistances()
        #self.init_kdistances()
        self.restriction_to_list()
        self._lambda = self.get_lambda_value()

    def local_search(self):
        #self.get_initial_solution()
        self.ls_solution = self.copkm().copy()
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
                #print('\n***Iteration***'+str(iteration))
                #print(self.ls_solution)
                # cluster from actual solution -> random access using neighbours index
                actual_index = self.neighbours[random_access[i]][0]
                actual_new_cluster = self.neighbours[random_access[i]][1]
                actual_cluster = self.ls_solution[actual_index]
                self.ls_solution[actual_index] = actual_new_cluster # (i0,i1) -> (index, cluster)

                if self.validate_actual_solution() == 1:
                    self.update_changed_centroids(actual_cluster, actual_new_cluster)
                    neighbor_f = self.f()
                    iteration += 1
                    if neighbor_f < actual_f:
                        # new solution to expand; new environment creation
                        #self.update_centroids(actual_cluster, actual_new_cluster)
                        # add update carray
                        #self.update_carray()
                        actual_f = neighbor_f
                        self.neighbours_generation()
                        #random_access = list(range(0,len(self.neighbours)))
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
                    self.ls_solution[actual_index] = actual_cluster
                    if return_anterior_state == 1:
                        self.update_changed_centroids(actual_new_cluster, actual_cluster)
                        return_anterior_state = 0
                        #self.update_carray()

                    # add update eds

        print('\nEvaluations: '+str(iteration))

        # bmb return to get best solution from all 10 iterations
        return self.f(), self.ls_solution.copy()

    '''
        Generates neighbours like: (index, value) ; |neighbours| = n_instances*(k-1) ; all possible combinations less the actual
    '''
    def neighbours_generation(self):
        if len(self.neighbours) > 0:
            self.neighbours.clear()
        for i in range(self.n_instances):
            for j in range(self.k):
                if j != self.ls_solution[i]:
                    self.neighbours.append((i, j))  # pair(index, value)

    def update_carray(self):
        self.c_array.clear()
        self.create_carray()
        for i in range(self.n_instances):
                self.c_array[self.ls_solution[i]].append(i)

    def update_changed_centroids(self, old_cluster, new_cluster):
            av_old = [0] * self.features
            av_new = [0] * self.features

            for i in range(len(self.ls_solution)):
                if self.ls_solution[i] == old_cluster or self.ls_solution[i] == new_cluster :
                    for k in range(self.features):
                        if self.ls_solution[i] == old_cluster:
                            av_old[k] = av_old[k] + self.data[k][i]
                        elif self.ls_solution[i] == new_cluster:
                            av_new[k] = av_new[k] + self.data[k][i]
            for k in range(self.features):
                self.centroids[old_cluster][k] = av_old[k] / self.ls_solution.count(old_cluster)
                self.centroids[new_cluster][k] = av_new[k] / self.ls_solution.count(new_cluster)

            # update distances(only related clusters)
            for i in range(self.n_instances):
                for j in range(self.k):
                    if j == old_cluster or j == new_cluster:
                        dist = 0
                        for k in range(self.features):
                            dist = dist + (self.data[k][i]-self.centroids[j][k]) **2
                        self.k_distances[i][j] = math.sqrt(dist)

    def get_general_deviation(self):
        mean_distance = [0] * self.k

        for i in range(self.n_instances):
            mean_distance[self.ls_solution[i]] += self.k_distances[i][self.ls_solution[i]]

        general_deviation = 0
        for k in range(self.k):
             mean_distance[k] /= self.ls_solution.count(k)
             general_deviation += mean_distance[k]
        general_deviation /= self.k
        return general_deviation

    def get_infeasibility(self):
        inf = 0
        for i in range(len(self.restrictions_list)):
            c1 = self.ls_solution[self.restrictions_list[i][0]]
            c2 = self.ls_solution[self.restrictions_list[i][1]]
            rest = self.restrictions_list[i][2]
            if rest == 1 and c1 != c2:
                inf += 1
            elif rest == -1 and c1 == c2:
                inf += 1
        return inf

    def get_initial_solution(self):
        self.create_carray()
        for i in range(self.n_instances):
            r_k = random.randint(0,self.k-1)
            self.c_array[r_k].append(i)
            self.ls_solution.append(r_k)
        while self.validate_actual_solution() == 0:
            self.ls_solution.clear()
            self.c_array.clear()
            self.create_carray()
            for i in range(self.n_instances):
                r_k = random.randint(0,self.k-1)
                self.c_array[r_k].append(i)
                self.ls_solution.append(r_k)
        self.update_centroids_ls()
        #update distances
        for i in range(self.n_instances):
            for j in range(self.k):
                dist = 0
                for k in range(self.features):
                    dist = dist + (self.data[k][i]-self.centroids[j][k]) **2
                self.k_distances[i].append(float(dist))

    def update_centroids_ls(self):
      for i in range(self.k):
          av = [0] * self.features
          for j in range(len(self.c_array[i])):
              for k in range(self.features):
                  av[k] = av[k] + self.data[k][self.c_array[i][j]]
              #if (len(self.c_array[i])) != 0:
              for k in range(self.features):
                  self.centroids[i][k] = av[k] / len(self.c_array[i])

    def validate_actual_solution(self):
        count = Counter(self.ls_solution)
        for i in range(self.k):
            if count[i] == 0:
                return 0
        return 1

    def correct_actual_solution(self):
        correct = 0

        while not correct:
            validation = self.k
            for j in range(self.k):
                if self.ls_solution.count(j) == 0: # if empty cluster
                    correct = 0
                    cluster = j
                    validation -= 1
            if validation == self.k:
                correct = 1

            if not correct:
                for i in range(self.n_instances):
                    if self.ls_solution[self.rsi[i]] != cluster and self.ls_solution.count(self.ls_solution[self.rsi[i]]) > 1:
                        self.ls_solution[self.rsi[i]] = cluster

    def show_clusters(self, algorithm):
        show_0 = []
        show_1 = []
        show_2 = []

        if len(self.ls_solution) > 0:
            for i in range(len(self.ls_solution)):
                self.c_array[self.ls_solution[i]].append(i)

        for i in range(2):
            show_0.append([])
            for j in range(len(self.c_array[0])):
                show_0[i].append(self.data[i][self.c_array[0][j]])
        plt.scatter(show_0[0], show_0[1])

        for i in range(2):
            show_1.append([])
            for j in range(len(self.c_array[1])):
                show_1[i].append(self.data[i][self.c_array[1][j]])
        plt.scatter(show_1[0], show_1[1])

        for i in range(2):
            show_2.append([])
            for j in range(len(self.c_array[2])):
                show_2[i].append(self.data[i][self.c_array[2][j]])
        plt.scatter(show_2[0], show_2[1])

        tmp = []
        for i in range(self.k):
            tmp.append([])

            tmp_x = self.centroids[i][0]
            tmp_y = self.centroids[i][1]
            plt.scatter(tmp_x, tmp_y, marker='^')

        #plt.show()
        #plt.savefig(self.save_file+'.png')
        plt.savefig(self.get_output_file(self.save_file+algorithm, 'png'))
        plt.close()
