import numpy as np
import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt


#Define states and transition matrix
states = ["Active","Idle","Inactive"]
transition_matrix = np.array([
    [0.3,0.5,0.2],
    [0.1,0.7,0.2],
    [0.05,0.1,0.85]
])

#Display the transition matrix
matrix_prob = pd.DataFrame(transition_matrix,index=states,columns=states)
print("Transition Matrix:")
print(matrix_prob)

#Display state options for user input
print("\n 0-'Active'\n 1-'Idle'\n2-'Inactive'")

#user input for intial state
intial_state = int(input("\nEnter the intial state (0,1, or 2): "))

#create the initial probabilities
state_prob=[0,0,0]
state_prob[intial_state]=1
initial_state_probs = np.array(state_probs)

#Number of days to simulate
nth_day = int(input("\nEnter the number of days to stimulate: "))

#Define the final state of interest
final_state = int(input("\nEnter the final state of interest (0,1, or 2): "))

#create a Directed Graph to represent the Markov chain
G=nx.DiGraph()

#Add nodes(states) to the graph 
G.add_nodes_from(states)

#simulate user engagement for each data and construct the Markov chain
current_state_probs = initial_state_probs

for day in range(nth_day):
    #get the next state based on current state and transition matrix
    next_state_probs = np.dot(current_state_probs,transition_matrix)

    #update the current state probabilities
    current_state_probs = next_state_probs

    print(f"\nDay {day+1} - state Probabilities: ")
    for i in range(len(states)):
        print(f"Probability of {states[i]}:{next_state_probs[i]:.2f}")

#update the Markov chain edges with transition probabilites
for i,state in enumerate(states):
    for j,next_state in enumerate(states):
        G.add_edge(state,next_state,probability=transition_matrix[i][j])

#calulate and display the final state probability 
final_state_probability = current_state_probs[final_state]
print(f"\nPrbability of being in state '{state[final_state]}' on the {nth_day}-th day:{final_state_probability:.2f}")

#Draw the Markov Chain diagram
pos = nx.spring_layout(G,seed=42)

#Define edge labels based on transition probabilities 
edge_labels = {(u,v):f"{G[u][v]['Probability']:.2f}" for u,v in G.edges()}

#Draw the Markov chain with a new color combination
nx.draw(G,pos,with_labels=True,node_size=1000,node_color='skyblue',font_size=9,font_color='black',font_weight='normal',arrows=True,arrowsize=15,linewidths=2,connectionstyle='arc3,rad=0.2')


#position edge labels for better visibility
label_pos = {edge:(0.5*(pos[edge[0]][0]+0.7*pos[edge[1]][0]),0.5*(0.7*pos[edge[0]][1]+pos[edge[1]][1]))for edge in G.edges()}

#Draw edge labels with specified positions
nx.draw_networkx.labels(G,label_pos,labels=edge_labels,font_color='red',font_size=13,font_weight='bold')

#show the plot
plt.title("Markov Chian for User Engagement")
plt.show()

