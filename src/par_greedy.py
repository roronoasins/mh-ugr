import random
from collections import Counter

from par_class import Par

class Greedy(Par):

    def __init__(self, f_name, n_rest, k, seed):
        Par.__init__(self, f_name, n_rest, k, seed)
        self._lambda = self.get_lambda_value()
        self.generate_centroids()
        self.create_kdistances()
        self.init_kdistances()

    '''
    COPKM(Greedy) algorithm

    - Se generan k centroides iniciales de forma aleatoria dentro del dominio asociado
    a cada dimensión del conjunto de datos.

    - Se barajan los índices de las instancias para recorrerlas de forma aleatoria sin
    repetición

    - Se asigna cada instancia del conjunto de datos al grupo más cercano (aquel con
    centroide más cercano) que no viole ninguna restricción ML ó CL. En caso de no
    existir, se devuelve la partición vacía y se termina el algoritmo.

    - Se actualizan los centroides de cada grupo de acuerdo al promedio de los valores
    de las instancias asociadas a su grupo.

    - Se repiten los dos pasos anteriores mientras que al menos un grupo haya sufrido
    algún cambio.
    '''

    def copkm(self):
        # random shuffle for initial case
        self.rsi = list(range(0,len(self.data[0])))
        # barajamos las instancias para recorrerlas de forma aleatoria
        random.shuffle(self.rsi)
        self.n_instances = len(self.rsi)
        self.create_closet_cluster()

        changed = 1
        i = 0
        while changed == 1:
            changed = 0
            i = i + 1
            if self.pick_centroid() == 1:
                changed = 1
            self.update_centroids()

            #print('\n\t***Iteration '+str(i)+'***\n')
            #self.print_centroids()
            #self.print_carray()
        self.update_centroids()
        print('Iterations: ' + str(i))
        return self.closet_cluster.copy()

    '''
    Picks closest centroid
    '''
    def pick_centroid(self):
        self.update_distances()
        return self.update_closest_cluster()

    # Update the closest cluster(each instance)
    def update_closest_cluster(self):
        changed = 0
        for i in range(self.n_instances):
            random_access = self.rsi[i]
            local_iv = []
            min_ci = []
            distances = []
            min_index = []
            min_d = 999
            min_i = self.closet_cluster[random_access]
            for j in range(self.k):
                iv = self.infeasibility(j, random_access)
                local_iv.append(iv)
                min_ci.append(j)
                distances.append(self.k_distances[random_access][j])
            smallest = min(local_iv)
            # if more than one with minimal iv, si no fufa probar añadir si <= a lista y si es < borrar lista y seguir. Luego comprobar si len > 1, si es asi comprobar dist de los que contenga la lista
            for index, element in enumerate(local_iv):
                if self.check_rest(random_access, min_ci[index]):
                    if smallest == element:  # check if this element is the minimum_value
                        min_index.append(index)  # add the index to the list if it is
            for a in range(len(min_index)):
                min_tmp = distances[min_index[a]]
                if min_tmp < min_d:
                    min_d = min_tmp
                    min_i = min_index[a]
            # if old != new: changed = 1; update c_array state
            if self.closet_cluster[random_access] != min_i:
                if self.closet_cluster[random_access] != 999:
                    self.c_array[self.closet_cluster[random_access]].remove(random_access)
                self.c_array[min_i].append(random_access)
                self.closet_cluster[random_access] = min_ci[min_i]
                changed = 1

        return changed

    def check_rest(self, index, new_cluster):
        valid = 1
        count = Counter(self.closet_cluster)
        if count[999] > 0:
            return 1
        old = self.closet_cluster[index]
        self.closet_cluster[index] = new_cluster
        count = Counter(self.closet_cluster)
        for i in range(self.k):
            if count[i] == 0:
                valid = 0
        self.closet_cluster[index] = old
        return valid

    '''
    Returns the number of restrictions
    '''
    def infeasibility(self, ci, selected):
        infeasibility_value = 0
        # check Cannot Links(CL)
        for i in range(len(self.c_array[ci])):
            if self.restrictions[selected][self.c_array[ci][i]] == -1:
                infeasibility_value += 1

        for i in range(self.n_instances):
            if (self.restrictions[selected][i] == 1) and (self.closet_cluster[selected] != self.closet_cluster[i]) and (self.closet_cluster[i] != 999 or self.closet_cluster[i] != 999):
                infeasibility_value += 1

        return infeasibility_value


    def get_infeasibility(self):
        infeasibility_value = 0
        diagonal = 1
        for i in range(self.n_instances):
            j = diagonal
            while j < self.n_instances:
                if (self.restrictions[i][j] == 1 and self.closet_cluster[i] != self.closet_cluster[j]) or (self.restrictions[i][j] == -1 and self.closet_cluster[i] == self.closet_cluster[j]):
                    infeasibility_value += 1
                j += 1
            diagonal += 1
        return infeasibility_value
