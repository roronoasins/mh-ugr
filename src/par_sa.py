from par_class import Par
from par_ag import AG

from numpy import log as ln
from collections import Counter
import random
import math

# Simulated Annealing class
class SA(AG):

  S = []
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
  boltzmann_k = 0.00000000000000013806 # boltzmann constant

  def __init__(self, f_name, n_rest, k, seed):
      Par.__init__(self, f_name, n_rest, k, seed)
      self.generate_centroids()
      self.create_kdistances()
      self.restriction_to_list()
      self._lambda = self.get_lambda_value()
      self.max_neighbours = 10*self.n_instances
      self.max_success = 0.1*self.max_neighbours
      self.M = self.max_evaluations/self.max_neighbours
      #print("Initial temperature: " + str(self.To))
      print("Final temperature: " + str(self.Tf))
      print("M: " + str(self.M))
      #self.check_Tf()
      # prob to accept worse solution than actual dependes on temperature and how much worse f() is

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
          print("Tf:"+str(self.Tf))
          f_inherited = False
          while i < self.max_neighbours and continue_cooling == False and evaluations < self.max_evaluations:
              # candidate solution (S')
              evaluations += 1
              f_s = self.f()
              candidate_s = self.get_neighbour() # hacer metodo para devolver S' y otro que sea f(solution) que calcule f en base a la estructura S pasado como arg
              evaluations += 1
              aux_s = self.S
              self.S = candidate_s.copy()
              self.update_carray()
              self.update_centroids() #fix
              self.update_distances()
              #print(self.validate_actual_solution())
              f_candidate_s = self.f()
              inc_f = f_candidate_s - f_s
              #print("candidate:"+str(f_candidate_s))
              #print("s:"+str(f_s))
              #print("inc:"+str(inc_f))
              #print("best:"+str(self.best_f))
              a = -inc_f/(T)
              #print(math.exp(a))
              if inc_f < 0 :#or random.random() <= math.exp(a):
                  f_s = f_candidate_s
                  if f_s < self.best_f:
                      self.best_f = f_s
                      best_s = self.S.copy()
                      success += 1
              else:
                  self.S = aux_s
                  self.update_carray()
                  self.update_centroids()
                  self.update_distances()
              i+=1
              if success == self.max_success:
                  print("reset")
                  continue_cooling = True

              print("Neigh: " + str(i))

          continue_cooling = False
          if success == 0:
              print("f")
              stucked = True
          #T = self.get_next_temperature_cauchy(T)
          T = self.get_next_temperature_lineal(T)
          print("T:"+str(T))
          print("Tf:"+str(self.Tf))

      #update data estructures with best_S
      self.S = best_s.copy()
      self.update_carray()
      self.update_centroids()
      self.update_distances()
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

  def f(self):
      return self.get_general_deviation_s() + (self.get_infeasibility_rl()*self._lambda)

  def update_carray(self):
      self.c_array.clear()
      self.create_carray()
      for i in range(self.n_instances):
              self.c_array[self.S[i]].append(i)

  def get_neighbour(self):
      neigh_S = self.S.copy()
      chromosome = random.randint(0, self.n_instances-1)
      k = neigh_S[chromosome]
      new_k = random.randint(0, self.k-1)
      while new_k == k:
          new_k = random.randint(0, self.k-1)
      neigh_S[chromosome] = new_k
      while self.validate_new_solution(neigh_S) == 0:
          # restore and try again
          '''
          neigh_S[chromosome] = k
          gen = random.randint(0, self.n_instances-1)
          k = neigh_S[chromosome]
          new_k = random.randint(0, self.k-1)
          while new_k == k:
              new_k = random.randint(0, self.k-1)
          '''
          chromosome = random.randint(0, self.n_instances-1)
          k = neigh_S[chromosome]
          new_k = random.randint(0, self.k-1)
          while new_k == k:
              new_k = random.randint(0, self.k-1)
          neigh_S[chromosome] = new_k
      return neigh_S.copy()

  def get_initial_solution(self):
      self.create_carray()
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

  def validate_actual_solution(self):
      count = Counter(self.S)
      for i in range(self.k):
          if count[i] == 0:
              return 0
      return 1

  def validate_new_solution(self, new_s):
      count = Counter(new_s)
      for i in range(self.k):
          if count[i] == 0:
              return 0
      return 1

  def check_factible(self, neigh):
      return

  def check_Tf(self):
      return

  def get_general_deviation_s(self):
      mean_distance = [0] * self.k

      for i in range(self.k):
          for j in range(len(self.c_array[i])):
              mean_distance[i] += self.k_distances[self.c_array[i][j]][i]

      general_deviation = 0
      for k in range(self.k):
           mean_distance[k] /= len(self.c_array[k])
           general_deviation += mean_distance[k]
      general_deviation /= self.k
      return general_deviation

  def get_infeasibility_rl(self):
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

  def update_c_array(self):
      self.c_array = [[]] * self.k
      for i in range(len(self.S)):
          self.c_array[S[i]].append(i)

  def print_solution(self):
      solution = [0] * self.n_instances
      for j in range(self.k):
          for i in range(len(self.c_array[j])):
              solution[self.c_array[j][i]] = j
      print('\n')
      print(solution)

  def print_ds(self, s, neigh_s):
      for i in range(self.n_instances):
          if s[i] != neigh_s[i]:
              print(s[i])
              print(neigh_s[i])
