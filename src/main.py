import random
import sys
import time

from par_greedy import Greedy
from par_ls import LS
from par_ag import AG
from par_am import AM
from par_sals import SA
from par_bmb import BMB
from par_ils import  ILS
from par_fa import FA

data_file = sys.argv[1]
rest_file = sys.argv[2]
algorithm = sys.argv[3]


if data_file == 'iris' or data_file == 'rand' or data_file == 'newthyroid':
    k = 3
else:
    k = 8

print('\n** Dataset: ' + data_file + ', Rest_percent: ' + rest_file + ', k: ' + str(k) + ', Algorithm: ' + algorithm+' **')

seeds = [457958018911888190, 5036837746746340898, 1068418037881742573, 4231440123076266961, 323876236622255711]
seed = random.randrange(sys.maxsize)
seed = seeds[0]
#seed = 1436538138260555953
random.seed(seed)

i_time = time.time_ns() / (10 ** 9)
if algorithm == 'greedy':
    prueba = Greedy(data_file, rest_file, k, seed)
    prueba.copkm()
elif algorithm == 'bl':
    prueba = LS(data_file, rest_file, k, seed)
    prueba.local_search()
elif algorithm == 'ag':
    select_type = sys.argv[4]
    cross_type = sys.argv[5]
    print(select_type)
    print(cross_type)
    prueba = AG(data_file, rest_file, k, seed, algorithm, select_type, cross_type)
    prueba.ag()
elif algorithm == 'am':
    gen_freq = sys.argv[4]
    size_ls = sys.argv[5]
    print(gen_freq)
    print(size_ls)
    if len(sys.argv) >= 7:
        better = sys.argv[6]
        print(better)
    else:
        better = None
    prueba = AM(data_file, rest_file, k, seed, gen_freq, size_ls, better)
    prueba.am()
elif algorithm == 'sa':
    prueba = SA(data_file, rest_file, k, seed)
    prueba.sa()
elif algorithm == 'bmb':
    prueba = BMB(data_file, rest_file, k, seed)
    prueba.bmb()
elif algorithm == 'ils':
    if len(sys.argv) == 5:
        prueba = ILS(data_file, rest_file, k , seed, sys.argv[4])
    else:
        prueba = ILS(data_file, rest_file, k , seed, 'ls')
    prueba.ils()
elif algorithm == 'fa':
    prueba = FA(data_file, rest_file, k, seed, sys.argv[4], sys.argv[5])
    prueba.fa()
f_time = time.time_ns() / (10 ** 9)
#prueba.print_restriction()
#print(prueba.n_restrictions())

if algorithm == 'greedy':
    prueba.print_carray()
    prueba.print_solution()
    prueba.print_centroids()
    print('\nGeneral deviation: ' + str(prueba.get_general_deviation()))
    print('Infeasibility: ' + str(prueba.get_infeasibility()))
    print('lambda: ' + str(prueba.get_lambda_value()))
    print('Agr: ' + str(prueba.f()))
elif algorithm == 'bl' or algorithm == 'bmb' or algorithm == 'ils':
    prueba.print_S()
    prueba.print_solution()
    prueba.print_centroids()
    print('\nGeneral deviation: ' + str(prueba.get_general_deviation()))
    print('Infeasibility: ' + str(prueba.get_infeasibility()))
    print(prueba.c_array)
    #prueba.print_neighbours()
    print('lambda: '+str(prueba.get_lambda_value()))
    print('Agr: ' +str(prueba.f()   ))
elif algorithm == 'ag' or algorithm == 'am' or algorithm == 'fa':
    prueba.print_carray()
    prueba.print_solution()
    prueba.print_centroids()
    print('\nGeneral deviation: ' + str(prueba.solution_deviation()))
    print('Infeasibility: ' + str(prueba.solution_inf()))
    print('lambda: ' + str(prueba.get_lambda_value()))
    print('Agr: ' + str(prueba.solution_f()))
elif algorithm == 'sa':
    prueba.print_carray()
    prueba.print_solution()
    prueba.print_centroids()
    print('\nGeneral deviation: ' + str(prueba.get_general_deviation()))
    print('Infeasibility: ' + str(prueba.get_infeasibility()))
    print('lambda: ' + str(prueba.get_lambda_value()))
    print('Agr: ' + str(prueba.f()))
#elif algorithm == 'fa':
    #print(prueba.fireflies)
print('\nElapsed Time: ' + str(f_time - i_time))

#prueba.show_clusters(algorithm)
