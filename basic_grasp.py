import random
import graph as gr
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

#global values
assignment      = []

#--------------------------------------------reactive_grasp_for_MiMALBP-2-----------------------------------------------
def grasp(tasks_list, alpha, max_iterations, max_search, cycle_time):

    optimal_value   = 1000                  #best number of workstations
    best_solution   = []                    #best solution (assignment )
    alpha1_sol      = []

    for j in range(max_iterations):

        

        #build feasible solutions (not necessarly optimal)
        solution        = construction_phase(alpha, tasks_list)

        #searching for optimal solution using neighborhood search
        solution        = local_search_phase(solution, max_search)

        n_workstations         = objective_function(solution, cycle_time)

      
        #comparing soltuion in terms of objective value---------------
        if( n_workstations < optimal_value):

            best_solution       = solution
            optimal_value       = n_workstations

            alpha_best          = alpha
        #-------------------------------------------------------------

        if(solution not in alpha1_sol):
                alpha1_sol.append(solution)

    print("number_of workstations:", optimal_value)

    print("-----------------------------------------")
    print("number of solutions a1:",len(alpha1_sol))
   
    
    return best_solution, alpha1_sol

#--------------------------------------------------------------------------------------------------------------

#construction phase function
def construction_phase(alpha, tasks_list):
    
    solution        = []
    cl              = {}
    RCL             = {}

  

    copy_tasks_list         = tasks_list.copy()

    while(copy_tasks_list):

            #create the candidate list
            for task in copy_tasks_list:

                pred_list       = list(gr.G.predecessors(int(task)))

                if not any(item in  pred_list for item in copy_tasks_list):
                         cl[task] = gr.G.nodes[task]["task"+str(task)]
                   

            
            """
            #calculate the threshold value
            times_cl      = cl.values()
            
            #calculate the threshold value
            threshold       = min(times_cl) + alpha*(max(times_cl) - min(times_cl))
            """

            j = 0
            
            #create Restricted candidate list
            while(j<2):

                for task in cl:
                
                    #if(cl[task] <= threshold):
                    if(cl[task] == min(list(cl.values()))):
                            RCL[task]       = cl[task]
                            cl.pop(task)
                            break
                j = j + 1  
                        
           
            
            #select randomly a task from the restricted candidate list
            selected_task       = random.choice(list(RCL.keys()))

            #adding task to the partial solution
            solution.append(selected_task) 

            #delete selected task from cl and RPW_list
            copy_tasks_list.remove(selected_task)
                
            #cl.pop(selected_task)
            cl.clear()
                
            #clean the RCL
            RCL.clear()


    return  list(np.array(solution, dtype=int))

#local search function
def local_search_phase(solution, max_search):
    
    neighbor_solutions      = []
    neighbor                = []
    #initial_sol             = solution
    iteraion_crit           = 0
    counter                 = 1
    #for j in range(max_search):
    


    while(True):

        continue_v              = False

        while(counter <= max_search):

            sol_copy      = solution.copy()
            #search for neighbor solution
            neighbor      = random_swap(sol_copy)

            if(not list(neighbor) in neighbor_solutions):

                neighbor_solutions.append(list(neighbor))

            else:

                counter = counter + 1

        
     
        

        for i in neighbor_solutions:
        #compare the neighbor solution with the old solution(constructed solution)
            if(objective_function(i, cycle_time) < objective_function(solution, cycle_time)):

                      solution      = i
                      counter       = 1
                      continue_v    = True

        if(not continue_v):

                        return solution


    """
    while(counter <= max_search):
            
            n_solution = 0
            while(True):
                
                sol_copy      = solution.copy()
                #search for neighbor solution
                neighbor      = random_swap(sol_copy)
                
                if(not list(neighbor) in neighbor_solutions):

                        neighbor_solutions.append(list(neighbor))
                        n_solution = n_solution + 1
                        
                        if(n_solution == 10):
                            break
                else
    
                iteraion_crit       = iteraion_crit + 1
                
                if(iteraion_crit > 30):
                        break

    for i in neighbor_solutions:
    #compare the neighbor solution with the old solution(constructed solution)
        if(objective_function(i, cycle_time) < objective_function(best_solution, cycle_time)):

            best_solution       = i

    
    return  best_solution
    """

#objective function to evaluat found soutions
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



def random_swap(solution):
    
    #neighbor_solution       = []
 
    while(True):


        task1       = random.choice(solution)
        index1      = solution.index(task1)

        
        task2       = random.choice([task for task in solution if task != task1])
        index2      = solution.index(task2)
        

        x           = index1
        y           = index2
        key         = True

        if( index1 < index2):
                            
                        if(task1 not in gr.nx.nodes(gr.nx.bfs_tree(gr.G, task2, reverse=True))):
                                
                                for a in solution[index1+1:index2]:
                                    
                                    if(task1 in gr.nx.nodes(gr.nx.bfs_tree(gr.G, a, reverse=True)) or a in gr.nx.nodes(gr.nx.bfs_tree(gr.G, task2, reverse=True))):
                                        key     = False
                                        continue

                                if(key == True):
                                    
                                    solution[x]     = task2
                                    solution[y]     = task1
                                    break
                                

        else:

                            if(task2 not in gr.nx.nodes(gr.nx.bfs_tree(gr.G, task1, reverse=True))):
                            
                                for a in solution[index2+1:index1]:
                                    
                                    if(task2 in gr.nx.nodes(gr.nx.bfs_tree(gr.G, a, reverse=True)) or a in gr.nx.nodes(gr.nx.bfs_tree(gr.G, task1, reverse=True)) ):
                                        key     = False
                                        break
                                
                                if(key == True):
                                    
                                    solution[x]     = task2
                                    solution[y]     = task1
                                    break




    return solution

    

#tasks_list      =[1,2,3,4,5,6,7,8,9,10]
#tasks_list      =[1,2,3,4,5,6,7,8,9,10,11,12]
#tasks_list      =[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#tasks_list      = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
tasks_list      = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
alpha           = 0
max_iterations  = 2000
max_search      = 40
cycle_time      = 7.8

start_time = datetime.now()
solution, s1= grasp(tasks_list, alpha, max_iterations, max_search, cycle_time)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

print(solution)







"""
sol     = construction_phase(1, tasks_list)
sol     = list(np.array(sol, dtype=int))

print(sol)
print(sol[2:4])

#print(random_swap(sol))
#print(gr.nx.nodes(gr.nx.bfs_tree(gr.G, 5, reverse=True)))
#print("its cycle time:", objective_function(sol, 3))

list1 = RPW(tasks_list)
#print(list1)
cl = {}
for task in list1:

    print(list(gr.G.predecessors(int(task))))
    pred_list       = np.array(list(gr.G.predecessors(int(task))), dtype=str)

    #any(item in list(gr.G.predecessors(a)) for item in CL)
    if not any(item in  pred_list for item in list1):
        cl[task]        =  list1[task]

print(random.choice(list(cl.keys())))
#print(random.choice(cl.ite))
"""