from par_ls import LS

class BMB(LS):
    max_evaluations = 10000
    max_iterations = 10


    def __init__(self, f_name, n_rest, k, seed):
        LS.__init__(self, f_name, n_rest, k, seed)

    def bmb(self):
        best_solution = []
        best_solution_f = 999999

        i = 0
        while i < self.max_iterations:
            f, S = self.local_search()
            if f < best_solution_f:
                best_solution_f = f
                best_solution = S.copy()
            print("Iteration "+str(i)+" completed")
            i+=1
            self.S.clear()

        self.S = best_solution.copy()
        self.update_carray()
        self.update_centroids()
        self.update_distances()
