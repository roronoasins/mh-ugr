from par_sals import SA

import random

class ILS(SA):
    iterations = 10
    max_evaluations = 10000
    best_f = 999999
    best_s = []

    def __init__(self, f_name, n_rest, k, seed):
        SA.__init__(self, f_name, n_rest, k, seed)
        self.mutation_value = 0.1*self.n_instances
