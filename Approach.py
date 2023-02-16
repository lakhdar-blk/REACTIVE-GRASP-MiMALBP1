import random
import graph as gr
import numpy as np
from datetime import datetime

assignment      = []

def appraoch_based_heuristics(tasks_list):

    solution            = []
    list_candidate      = []
    list_candidate2     = []
    list_candidate3     = []
    n_successors        = {}
    maximum_task_time   = {}

    tmp_list1 = []
    
    for task in tasks_list:
        n_successors[str(task)]      = len(list(gr.nx.nodes(gr.nx.dfs_tree(gr.G, task)))) - 1

    for task in tasks_list:
        maximum_task_time[str(task)] = gr.G.nodes[task]["task"+str(task)]

    

    #ranked poistoinal weight
    RPW_l             = RPW(tasks_list)


    def add_task(task):
        


        solution.append(task)
        RPW_l.pop(task)
        n_successors.pop(task)
        maximum_task_time.pop(task)


        list_candidate.clear()
        list_candidate2.clear()
        list_candidate3.clear()

    while(RPW_l):

        for i in RPW_l.keys():
            if(RPW_l[i] == max(RPW_l.values())):
                    list_candidate.append(i)

        if(len(list_candidate) > 1):
           
            for i in list_candidate:
                tmp_list1.append(n_successors[str(i)])
            
            for i in list_candidate:                  
                if(n_successors[str(i)] == max(tmp_list1)):
                    list_candidate2.append(i)
            
            tmp_list1.clear()

            if(len(list_candidate2) > 1):

                for i in list_candidate2:
                    tmp_list1.append(maximum_task_time[str(i)])


                for i in list_candidate2:
                    if(maximum_task_time[str(i)] == max(tmp_list1)):
                        list_candidate3.append(i)

                tmp_list1.clear()

                if(len(list_candidate3) > 1):
                    task = random.choice(list_candidate3)
                    add_task(task)

                else:
                 
                    add_task(list_candidate3[0])
            else:
                add_task(list_candidate2[0])

        else:
            add_task(list_candidate[0])
            


    
    
    return solution

#Ranked Positional Weight
def RPW(tasks_list2):
    
    RPW_list            = {}
    highest_pw          = 0
    positional_weight   = 0
    tasks_list          = tasks_list2.copy()

    

    while(tasks_list):
        
        for task in tasks_list:

                suc_list        = gr.nx.nodes(gr.nx.dfs_tree(gr.G, task))

                for i in suc_list:

                    positional_weight       = positional_weight + gr.G.nodes[i]["task"+str(i)]
                    


                if(positional_weight > highest_pw):

                    highest_pw      = positional_weight
                    t               = task
                
                positional_weight       = 0
       
        #RPW_list.append(t)
        RPW_list[str(t)]        = highest_pw
        tasks_list.remove(t)
        
        highest_pw      = 0
    #------------------------------------------------------------------------------

    return RPW_list

def objective_function(solution, cycle_time):
    
    number_of_workstations              = 1
    w                                   = []
    tmp_cycle_time                      = cycle_time
    int_solution                        = list(np.array(solution, dtype=int))
    int_solution2                       = int_solution.copy()

    

    while(int_solution):

        for task in int_solution2:

          

            if(gr.G.nodes[task]["task"+str(task)] <= tmp_cycle_time):
                   
                    w.append(task)
                    tmp_cycle_time      = tmp_cycle_time - gr.G.nodes[task]["task"+str(task)]
                    int_solution.remove(task)

                    if(not int_solution):
                        assignment.append(w)

            else:
                   
                    assignment.append(w)
                    w       = []
                    number_of_workstations = number_of_workstations + 1
                    tmp_cycle_time = cycle_time
                    w.append(task)
                    tmp_cycle_time      = tmp_cycle_time - gr.G.nodes[task]["task"+str(task)]
                    int_solution.remove(task)
       
    return number_of_workstations

#tasks_list = [1,2,3,4,5,6,7,8,9,10]
#tasks_list      =[1,2,3,4,5,6,7,8,9,10,11,12]
#tasks_list      =[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#tasks_list      = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
tasks_list      = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
c           = 7.8

start_time = datetime.now()
solution    = appraoch_based_heuristics(tasks_list)
print("objective value : ", objective_function(solution, c))
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
print(solution)

