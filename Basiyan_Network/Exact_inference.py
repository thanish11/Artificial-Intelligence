#pip install pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import numpy as np

#define network structure
alarm_dependency_model = BayesianNetwork([
    ('Burglary','Alarm'),
    ('Earthquake','Alarm'),
    ('Alarm','Marycalls'),
    ('Alarm','Johncalls')
])
#define the CPDs
cpd_burglary=TabularCPD(variable='Burglary',variable_card=2,values=[[0.999],[0.001]])
cpd_earthquake=TabularCPD(variable='Earthquake',variable_card=2,values=[[0.998],[0.002]])
cpd_alarm = TabularCPD(variable='Alarm', variable_card=2,
                       values=[[0.999, 0.71, 0.06, 0.05],  # P(Alarm | Burglary=False, Earthquake=False)
                               [0.001, 0.29, 0.94, 0.95]],  # P(Alarm | Burglary=True, Earthquake=True)
                       evidence=['Burglary', 'Earthquake'],
                       evidence_card=[2, 2])
cpd_marycalls=TabularCPD(variable='Marycalls',variable_card=2,
                        values=[[0.95,0.10], # P(MaryCalls | Alarm=False)
                               [0.05,0.90]], # P(MaryCalls | Alarm=True)
                         evidence=['Alarm'],
                         evidence_card=[2])
cpd_johncalls=TabularCPD(variable='Johncalls',variable_card=2,
                        values=[[0.99,0.30],
                               [0.01,0.70]],
                        evidence=['Alarm'],
                        evidence_card=[2])
# Add CPDs to the model
alarm_dependency_model.add_cpds(
    cpd_burglary,
    cpd_earthquake,
    cpd_alarm,
    cpd_johncalls,
    cpd_marycalls
)
# Create a directed graph from the Bayesian network structure
import networkx as nx
import matplotlib.pyplot as plt
graph = nx.DiGraph()
graph.add_edges_from(alarm_dependency_model.edges())

# Draw the graph
#pos = nx.spring_layout(graph, seed=42)
pos = pos={'Burglary':(1,5),
    'Earthquake':(5,5),
    'Alarm':(2.5,2.5),
    'Marycalls':(4,1),
    'Johncalls':(1,1)}
nx.draw(graph, pos, with_labels=True, node_size=10000, node_color="skyblue", font_size=10)

# Display the graph
plt.title("Alarm Bayesian Network")
plt.show()

#cpd tables view
for node in alarm_dependency_model.nodes():
    cpd = alarm_dependency_model.get_cpds(node)
    print(f"CPD for {node}:\n{cpd}\n")

'''with this gona find the Exact Inference, the types of 
Exact Inference are Variable enumeration and Variable Elimination'''

#Enumeration
cpd_values = {}
for node in alarm_dependency_model.nodes:
    cpd_tab = alarm_dependency_model.get_cpds(node)
    cpd_values[node] = cpd_tab.values

jp = [0, 0]

for c in range(len(cpd_values['Burglary'])):
    sum_eq = 0
    for a in range(len(cpd_values['Earthquake'])):
        sum_alarm = 0
        for b in range(len(cpd_values['Alarm'])):
            sum_alarm += cpd_values['Alarm'][b][c][a] * cpd_values['Johncalls'][1][b] * cpd_values['Marycalls'][1][b]
        sum_eq += sum_alarm * cpd_values['Earthquake'][a]
    jp[c] += sum_eq * cpd_values['Burglary'][c]

cp = jp / np.sum(np.array(jp))
#print(jp)
#print(cp)
print("\nVARIABLE ENUMERATION")
print(f"\nP('Burglary' / 'johncalls'= True , 'marycalls'=True) = {cp}")


#Elimination
from pgmpy.inference import VariableElimination
infer = VariableElimination(alarm_dependency_model)
probability = infer.query(['Burglary'],evidence={'Johncalls':1,'Marycalls':1})
print("\nVARIABLE ELIMINATION")
print(f"\nP('Burglary' / 'johncalls'= True , 'marycalls'=True)  \n{probability}")