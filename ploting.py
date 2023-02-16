import matplotlib.pyplot as plt

x1 = ['Hbyrid reactive GRASP', 'Basic GRASP', 'Approach based heuristics']
v1 = [5, 5, 5]
v2 = [4, 4, 5]
v3 = [6, 7, 7]
v4 = [7, 7, 8]
v5 = [8, 9, 9]
v6 = [7, 7, 7]
#plt.figure(figsize=(3, 6))

"""
plt.subplot(311)
plt.bar(x, p1, c = 'b', label = "alpha = 0")
plt.subplot(312)
plt.bar(x, p2, c = 'g', label = "alpha = 0.5")
plt.subplot(313)
plt.bar(x, p3, c = 'r' , label = "alpha = 1")
"""
plt.bar(x1, v6, color = ['blue', 'green', 'red'])

plt.xlabel('Algorithms')
plt.ylabel('Number of workstations')
plt.suptitle('Probabilities of selection')

plt.legend() 
plt.show()