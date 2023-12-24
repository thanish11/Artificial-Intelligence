#Approximate Inferece - Direct Sampling and Gibbs Sampling

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import numpy as np

#define network structure 
direct_models=BayesianNetwork([
    ('Cloudy', 'Rain'), ('Cloudy', 'Sprinkler'), ('Sprinkler','Wetgrass'), ('Rain', 'Wetgrass')
])

#define CPDs
cpd_cloudy = TabularCPD(variable='Cloudy', variable_card=2, values=[[0.5], [0.5]])
cpd_rain = TabularCPD(variable='Rain', variable_card=2,
values=[[0.8, 0.2], [0.2, 0.8]],
evidence=['Cloudy'],
evidence_card=[2])

cpd_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2,
values=[[0.5, 0.9], [0.5, 0.1]],
evidence=['Cloudy'],
evidence_card=[2])

cpd_wetgrass = TabularCPD(variable='Wetgrass', variable_card=2,
values=[[0.01, 0.1, 0.1, 1.0],
[0.99, 0.90, 0.90, 0.00]],
evidence=['Sprinkler', 'Rain'],
evidence_card=[2, 2])

direct_models.add_cpds(cpd_cloudy, cpd_rain, cpd_sprinkler, cpd_wetgrass)

#add the cpds to modles
for node in direct_models.nodes():
    cpd = direct_models.get_cpds(node)
    print(f"CPD for {node}:\n{cpd}\n")

#Approximate Inference --Direct Sampling
cpd_value={}
for nodes in direct_models.nodes:
  cpd_tab=direct_models.get_cpds(nodes)
  cpd_value[nodes]=cpd_tab.values


jps={}
for i in range(len(cpd_value['Cloudy'])):
  for j in range(len(cpd_value['Sprinkler'])):
    for k in range(len(cpd_value['Rain'])):
      for l in range(len(cpd_value['Wetgrass'])):
        jps[(i,j,k,l)] = cpd_value['Cloudy'][i] * cpd_value['Sprinkler'][j][i] * cpd_value['Rain'][k][i] * cpd_value['Wetgrass'][l][j][k]
direct_sampling = jps[(1,0,1,1)]




#Inference by Markov chain simulation  --Markov chain Monte Carlo (MCMC) --Gibbs Sampling

from pgmpy.sampling import GibbsSampling
import random
gibbs = GibbsSampling(direct_models)
sample = gibbs.sample(size=1000)

jp_count= len(sample[(sample['Cloudy']==1)&(sample['Sprinkler']==0)&(sample['Rain']==1)&(sample['Wetgrass']==1)])
jp_s=jp_count/len(sample)
print(jp_s)


