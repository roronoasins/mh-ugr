#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 21:38:43 2020

@author: roronoasins
"""

import random
import matplotlib.pyplot as plt
import math
import time
import sys
import numpy as np

class Par(object):
    '''
    Clase para manejar el Problema del Agrupamiento con Restricciones
    '''

    data = []
    restrictions = []
    restrictions_list = []
    centroids = []
    k = 0
    # random shuffle
    rsi = []
    n_instances = 0
    k_distances = []
    closet_cluster = []
    # k-dimension array with actual k-cluster state
    c_array = []
    features = 0
    solution = 1
    seed = 0
    # solution
    S = []

    def __init__(self, f_name, n_rest, k, seed):
        # seed generation
        self.seed = seed
        random.seed(self.seed)
        print("\nUsed seed:", seed)
        # COMMON
        self.file_name = '../instancias/'+f_name+'_set.dat'
        self.r_file = '../instancias/'+f_name+'_set_const_'+str(n_rest)+'.const'
        self.save_file = 'graphics/'+f_name+'-'+str(n_rest)
        self.k = k
        self.features = self._dimension()
        self._data()
        self.n_instances = len(self.data[0])
        self._restriction()
        self.create_carray()
        self.rsi = list(range(0, self.n_instances)) # mirar si solo es de greedy(y si es asi meterlo en par_greedy)
        # barajamos las instancias para recorrerlas de forma aleatoria
        random.shuffle(self.rsi)

    '''
    matrix[n_instances][k] -> distance between ni and ci
    row                      -> -- distance_c0 ... distance_ci ci ---
    '''
    def create_kdistances(self):
        for i in range(self.n_instances):
            self.k_distances.append([])

    '''
    array[k] = [[]i, ..., []k] witha ctual state(k-cluster, ci)
    '''
    def create_carray(self):
        for i in range(self.k):
            self.c_array.append([])

    '''
    array[n_instances] (empty)
    '''
    def create_closet_cluster(self):
        for i in range(self.n_instances):
            self.closet_cluster.append(999)

    '''
    Dimension = |features|
    '''
    def _dimension(self):
        f = open(self.file_name, "r+")
        line = f.readline().split(',')
        f.close()
        return len(line)

    '''
    Get data from input file. Data = [[]i, ..., []k] / k = dimension
    '''
    def _data(self):
        for x in range(self.features):
            self.data.append([])
        with open(self.file_name) as data_file:
            # saves data instances in each line from input file
            for line in data_file:
                instancia = line.split(",")
                # each dimension is saved in specific slot
                for i in range(self.features):
                    self.data[i].append(float(instancia[i]))

    '''
    Get restriction values from input file. m[i][j] -> restriction (i,j)
    '''
    def _restriction(self):
        for x in range(self.n_instances):
            self.restrictions.append([])
        with open(self.r_file) as rest_file:
            # saves restriction instances in each line from input file
            for line in rest_file:
                instancia = line.split(",")
                # each dimension is saved in specific slot
                for i in range(self.n_instances):
                    self.restrictions[i].append(int(instancia[i]))

    '''
    Returns the number of restrictions
    '''
    def n_restrictions(self):
        n = 0
        #cols = self.n_instances-1 # only check above diagonal
        x = 0
        y = 0
        diagonal = 1
        while x < self.n_instances:
            y = diagonal
            while y < self.n_instances:
                if (self.restrictions[x][y] == 1 or self.restrictions[x][y] == -1):
                    n += 1
                y += 1
            x += 1
            diagonal += 1
        return n

    '''
    Initial solution creation
    '''
    def get_initial_solution(self):
        #self.create_carray()
        for i in range(self.n_instances):
            r_k = random.randint(0,self.k-1)
            self.c_array[r_k].append(i)
            self.S.append(r_k)
        while self.validate_actual_solution() == 0:
            self.S.clear()
            self.c_array.clear()
            self.create_carray()
            for i in range(self.n_instances):
                r_k = random.randint(0,self.k-1)
                self.c_array[r_k].append(i)
                self.S.append(r_k)
        self.update_centroids()
        #update distances
        for i in range(self.n_instances):
            for j in range(self.k):
                dist = 0
                for k in range(self.features):
                    dist = dist + (self.data[k][i]-self.centroids[j][k]) **2
                self.k_distances[i].append(float(dist))

    def validate_new_solution(self, new_s):
        count = Counter(new_s)
        for i in range(self.k):
            if count[i] == 0:
                return 0
        return 1

    '''
    Generate initial random centroids within the associated domain
    '''
    def generate_centroids(self):
        for j in range(self.k):
            self.centroids.append([])
            for i in range(self.features):
                self.centroids[j].append(random.uniform(0, max(self.data[i])))

    def init_kdistances(self):
        for i in range(self.n_instances):
            for j in range(self.k):
                dist = 0
                for k in range(self.features):
                    dist += (self.data[k][i]-self.centroids[j][k]) **2
                self.k_distances[i].append(math.sqrt(dist))


    '''
    Update methods
    '''
    # Update the distances between instances
    def update_distances(self):
        for i in range(self.n_instances):
            for j in range(self.k):
                dist = 0
                for k in range(self.features):
                    dist = dist + (self.data[k][i]-self.centroids[j][k]) **2
                #self.k_distances[i].append(float(dist))
                self.k_distances[i][j] = math.sqrt(dist)

    # Update centroids based in data within clusters
    def update_centroids(self):
        for i in range(self.k):
            av = [0] * self.features
            for j in range(len(self.c_array[i])):
                for k in range(self.features):
                    av[k] = av[k] + self.data[k][self.c_array[i][j]]
                #if (len(self.c_array[i])) != 0:
                for k in range(self.features):
                    self.centroids[i][k] = av[k] / len(self.c_array[i])


    '''
    def update_c_array(self):
        self.c_array = [[]] * self.k
        for i in range(len(self.S)):
            self.c_array[S[i]].append(i)
    '''
    def update_carray(self):
        self.c_array.clear()
        self.create_carray()
        for i in range(self.n_instances):
                self.c_array[self.S[i]].append(i)

    def update_changed_centroids(self, old_cluster, new_cluster):
            av_old = [0] * self.features
            av_new = [0] * self.features

            for i in range(len(self.S)):
                if self.S[i] == old_cluster or self.S[i] == new_cluster :
                    for k in range(self.features):
                        if self.S[i] == old_cluster:
                            av_old[k] = av_old[k] + self.data[k][i]
                        elif self.S[i] == new_cluster:
                            av_new[k] = av_new[k] + self.data[k][i]
            for k in range(self.features):
                self.centroids[old_cluster][k] = av_old[k] / self.S.count(old_cluster)
                self.centroids[new_cluster][k] = av_new[k] / self.S.count(new_cluster)

            # update distances(only related clusters)
            for i in range(self.n_instances):
                for j in range(self.k):
                    if j == old_cluster or j == new_cluster:
                        dist = 0
                        for k in range(self.features):
                            dist = dist + (self.data[k][i]-self.centroids[j][k]) **2
                        self.k_distances[i][j] = math.sqrt(dist)
            #self.update_carray()

    '''
        f = C + (infeasibility*λ); λ = D/|R|; D = max distance in dataset
    '''
    def f(self):
        return self.get_general_deviation() + (self.get_infeasibility2()*self._lambda)

    def get_lambda_value(self):
        return self.get_max_distance() / self.n_restrictions()

    '''
    C = mean(intra-cluster deviation) ; intra-cluster deviation = mean(euq_dist(i, ci))
    '''
    def get_general_deviation(self):
        mean_distance = [0] * self.k

        for i in range(self.k):
            for j in range(len(self.c_array[i])):
                mean_distance[i] += self.k_distances[self.c_array[i][j]][i]

        general_deviation = 0
        for k in range(self.k):
            #if len(self.c_array[k]) > 0:
             mean_distance[k] /= len(self.c_array[k])
             general_deviation += mean_distance[k]
        general_deviation /= self.k
        return general_deviation

    '''
    Export restriction matrix to restriction list -> [[xi xj 1 | -1]] ; xi and xj are the instances linked. 1->ML, -1->CL
    '''
    def restriction_to_list(self):
        for i in range(self.n_instances):
            j = i + 1
            while j < self.n_instances:
                if ((self.restrictions[i][j] == 1) and (i != j)) or (self.restrictions[i][j] == -1):
                    self.restrictions_list.append([i,j,self.restrictions[i][j]])
                j += 1

    def get_infeasibility(self):
        infeasibility_value = 0
        diagonal = 1
        for i in range(self.n_instances):
            j = diagonal
            while j < self.n_instances:
                if (self.restrictions[i][j] == 1 and self.S[i] != self.S[j]) or (self.restrictions[i][j] == -1 and self.S[i] == self.S[j]):
                    infeasibility_value += 1
                j += 1
            diagonal += 1
        return infeasibility_value

    def get_infeasibility2(self):
        inf = 0
        for i in range(len(self.restrictions_list)):
            c1 = self.S[self.restrictions_list[i][0]]
            c2 = self.S[self.restrictions_list[i][1]]
            rest = self.restrictions_list[i][2]
            if rest == 1 and c1 != c2:
                inf += 1
            elif rest == -1 and c1 == c2:
                inf += 1
        return inf

    '''
        Check distance between instances(with no repetition)
    '''
    def get_max_distance(self):
        dist = 0
        max_dist = 0
        diagonal = 1
        for i in range(self.n_instances):
            j = diagonal
            while j < self.n_instances:
                for k in range(self.features):
                    dist += (self.data[k][i]-self.data[k][j]) **2
                dist = math.sqrt(dist)
                if dist > max_dist:
                    max_dist = dist
                else:
                    dist = 0
                j += 1
            diagonal += 1

        return max_dist

    def get_output_file(self, basename, ext):
        import itertools
        import os
        actualname = "%s.%s" % (basename, ext)
        c = itertools.count()
        while os.path.exists(actualname):
            actualname = "%s (%d).%s" % (basename, next(c), ext)
        return actualname

    '''
        Show methods hacer el metodo mirando la dimension actual del problema
    '''
    def show_clusters(self, algorithm):
        show_0 = []
        show_1 = []
        show_2 = []

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

    def show_centroids(self):
        for i in range(self.k):
            tmp_x = self.centroids[i][0]
            tmp_y = self.centroids[i][1]
        plt.scatter(tmp_x, tmp_y)
        plt.show()
        plt.close()

    '''
    Print methods
    '''
    def print_centroids(self):
        print('\nCentroids:')
        for i in range(self.k):
            print('c'+str(i)+': '+str(self.centroids[i]))

    def print_closest_cluster(self):
        for i in range(self.n_instances):
            print('n'+str(i)+': '+str(self.closet_cluster[i]))

    def print_data(self):
        print(len(self.data))

    def print_kdistances(self):
        print(self.k_distances)

    def print_carray(self):
        print('\nClusters data:\n')
        print(self.c_array)

    def print_restriction(self):
        print(self.restrictions)

    def print_rsi(self):
       print(self.rsi)

    def print_cselection(self):
        print(self.c_selection)

    def print_closet_cluster(self):
        print(self.closet_cluster[self.n_instances])

    def print_general_deviation(self):
        print('\nGeneral_deviation: '+str(self.get_general_deviation()))

    def print_S(self):
        print('\nSolution:')
        print(self.S)

    def print_solution(self):
        solution = [0] * self.n_instances
        for j in range(self.k):
            for i in range(len(self.c_array[j])):
                solution[self.c_array[j][i]] = j
        print('\n')
        print(solution)

    def print_rl(self):
        print(self.restrictions_list)

    def print_neighbours(self):
        print(self.neighbours)
