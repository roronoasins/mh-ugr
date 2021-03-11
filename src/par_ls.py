import math
import random
import sys
import matplotlib.pyplot as plt
from collections import Counter

from par_class import Par

class LS(Par):

    neighbours = []
    _lambda = 0
    max_evaluations = 100000

    def __init__(self, f_name, n_rest, k, seed):
        Par.__init__(self, f_name, n_rest, k, seed)
        self.generate_centroids()
        self.create_kdistances()
        self.restriction_to_list()
        self._lambda = self.get_lambda_value()

    def local_search(self):
        self.get_initial_solution()
        self.neighbours_generation()
        random_access = list(range(0,len(self.neighbours)))
        random.shuffle(random_access)

        actual_f = self.f()
        iteration = 0
        go_back = 0
        end = False
        while not end and iteration < self.max_evaluations:
            # each evaluation, flag restarted
            end = True
            for i in range(len(self.neighbours)):
                # cluster from actual solution -> random access using neighbours index
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
                    go_back = 0

        print('\nEvaluations: '+str(iteration))
        return self.f(), self.S.copy()

    '''
        Generates neighbours like: (index, value) ; |neighbours| = n_instances*(k-1) ; all possible combinations less the actual
    '''
    def neighbours_generation(self):
        if len(self.neighbours) > 0:
            self.neighbours.clear()
        for i in range(self.n_instances):
            for j in range(self.k):
                if j != self.S[i]:
                    self.neighbours.append((i, j))  # pair(index, value)



    def get_general_deviation(self):
        mean_distance = [0] * self.k

        for i in range(self.n_instances):
            mean_distance[self.S[i]] += self.k_distances[i][self.S[i]]

        general_deviation = 0
        for k in range(self.k):
            mean_distance[k] /= self.S.count(k)
            general_deviation += mean_distance[k]
        general_deviation /= self.k
        return general_deviation

    def validate_actual_solution(self):
        count = Counter(self.S)
        for i in range(self.k):
            if count[i] == 0:
                return 0
        return 1

    def correct_actual_solution(self):
        correct = 0

        while not correct:
            validation = self.k
            for j in range(self.k):
                if self.S.count(j) == 0: # if empty cluster
                    correct = 0
                    cluster = j
                    validation -= 1
            if validation == self.k:
                correct = 1

            if not correct:
                for i in range(self.n_instances):
                    if self.S[self.rsi[i]] != cluster and self.S.count(self.S[self.rsi[i]]) > 1:
                        self.S[self.rsi[i]] = cluster

    def show_clusters(self, algorithm):
        show_0 = []
        show_1 = []
        show_2 = []

        if len(self.S) > 0:
            for i in range(len(self.S)):
                self.c_array[self.S[i]].append(i)

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

        plt.savefig(self.get_output_file(self.save_file+algorithm, 'png'))
        plt.close()
