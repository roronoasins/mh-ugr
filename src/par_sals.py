from par_class import Par
from par_ls import LS

from numpy import log as ln
from collections import Counter
import random
import math

# Simulated Annealing class
class SA(LS):
  _lambda = 0
  T = 0
  To = 0 # initial temperature
  Tf = 0.001 # final temperature // check if smaller than initial temperature(To)
  max_neighbours = 0
  max_success = 0
  max_evaluations = 100000
  M = 0
  mu = 0.3
  phi = 0.3
  best_f = 0
  boltzmann_k = 0.00000000000000013806 # boltzmann const

  def __init__(self, f_name, n_rest, k, seed):
      Par.__init__(self, f_name, n_rest, k, seed)
      self.generate_centroids()
      self.create_kdistances()
      self.restriction_to_list()
      self._lambda = self.get_lambda_value()
      self.max_neighbours = 10*self.n_instances
      #self.max_neighbours = 5*self.n_instances
      #self.max_neighbours = self.n_instances
      self.max_success = 0.1*self.max_neighbours
      self.M = self.max_evaluations/self.max_neighbours
      print("Final temperature: " + str(self.Tf))
      print("M: " + str(self.M))

  def sa(self):
      self.get_initial_solution()
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
          while i < self.max_neighbours and continue_cooling == False and evaluations < self.max_evaluations:
              print(evaluations)
              aux_s = self.S.copy()
              f_s = self.f()
              evaluations += 1
              old_data = self.get_neighbour()
              old_cluster = old_data[1]
              new_cluster = self.S[old_data[0]]
              self.update_changed_centroids(old_cluster, new_cluster)
              #update c_array
              self.c_array[old_cluster].remove(old_data[0])
              self.c_array[new_cluster].append(old_data[0])
              f_candidate_s = self.f()
              evaluations += 1
              inc_f = f_candidate_s - f_s
              a = -inc_f/(T)
              if inc_f < 0 or random.random() <= math.exp(a):
                  f_s = f_candidate_s
                  if f_s < self.best_f:
                      self.best_f = f_s
                      best_s = self.S.copy()
                      success += 1
              # return past state, create method
              else:
                  self.S = aux_s.copy()
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
      print("Evaluations: "+str(evaluations))
      return best_s

  # mu -> So worsening(%) we accept
  def get_To(self):
      return (self.mu*self.f())/(-ln(self.phi))

  def get_next_temperature_cauchy(self, Tk):
      return Tk/(1+self.beta()*Tk)

  def get_next_temperature_lineal(self,Tk):
      return Tk*0.9

  def beta(self):
      return (self.To-self.Tf)/(self.M*self.To*self.Tf)

  def get_neighbour(self):
      gen = random.randint(0, self.n_instances-1)
      old_k = self.S[gen]
      new_k = random.randint(0, self.k-1)
      while new_k == old_k:
          new_k = random.randint(0, self.k-1)
      self.S[gen] = new_k
      while self.validate_actual_solution() == 0:
          self.S[gen] = old_k
          gen = random.randint(0, self.n_instances-1)
          old_k = self.S[gen]
          new_k = random.randint(0, self.k-1)
          while new_k == old_k:
              new_k = random.randint(0, self.k-1)
          self.S[gen] = new_k
      return gen, old_k

  def check_factible(self, neigh):
      return

  def check_Tf(self):
      return

  def get_centroids(self, S):
      # first generate local carray with S
      carray = [[]] * self.k
      for i in range(self.n_instances):
          carray[S[i]].append(i)

      # generate centroids that S would have
      centroids = [[]] * self.k
      for i in range(self.k):
          av = [0] * self.features
          centroids[i] = [0] * self.features
          for j in range(len(carray[i])):
              for k in range(self.features):
                  av[k] += self.data[k][carray[i][j]]
              for k in range(self.features):
                  centroids[i][k] = av[k] / len(carray[i])
      return centroids

  def get_distance_to_centroids(self, centroids): # (self, centroids)
      distances = []
      for i in range(self.n_instances):
          distances.append([])
          for j in range(self.k):
              dist = 0
              for k in range(self.features):
                  dist += (self.data[k][i] - centroids[j][k]) ** 2
              distances[i].append(math.sqrt(dist))
      return distances

  def print_ds(self, s, neigh_s):
      for i in range(self.n_instances):
          if s[i] != neigh_s[i]:
              print(s[i])
              print(neigh_s[i])
