#Approximate Inference --Rejection Sampling

from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

cpd_intelligence = TabularCPD(variable= 'Intelligence', variable_card= 2 ,
                              values=[[ 0.7],[0.3]])

cpd_difficulty = TabularCPD( variable='Difficulty',variable_card= 2 ,
                             values=[[ 0.6], [0.4]])

cpd_cgpa = TabularCPD(variable= 'CGPA', variable_card= 3,
                      values=[[0.3,0.4,0.2,0.35 ],[0.3,0.2,0.3,0.45],[0.4,0.4,0.5,0.2]] ,
evidence=['Intelligence','Difficulty' ] ,evidence_card=[2,2 ])

cpd_gre = TabularCPD(variable= 'GRE', variable_card= 2 ,values= [[0.8,0.2] ,
[0.2, 0.8] ] , evidence=[ 'Intelligence' ] , evidence_card =[2])

cpd_admit = TabularCPD( variable='Admit', variable_card = 2 ,
values=[[ 0.6,0.2,0.55,0.6,0.6,0.3],[0.4,0.8,0.45 ,0.4,0.4,0.7] ],evidence= ['CGPA','GRE' ], evidence_card= [3, 2] )

model = BayesianNetwork([('Intelligence', 'CGPA'),
('Difficulty', 'CGPA'),
('CGPA', 'Admit'),
('Intelligence', 'GRE'),
('GRE', 'Admit')])

model.add_cpds(cpd_intelligence,cpd_difficulty,cpd_cgpa,
cpd_gre,cpd_admit)

assert model.check_model()

# Create the BayesianModelSampling object
sampling = BayesianModelSampling(model)

# Rejection Sampling Function
def rejection_sampling(model, evidence, query_variable, n_samples):
    samples = []
    for _ in range(n_samples):
        sample = sampling.forward_sample(size=1).iloc[0]
        #s.append(sample)
        if all(sample[var] == value for var, value in evidence.items()):
            samples.append(sample[query_variable])
    return samples

# Define evidence and query variable
evidence = {'Admit': 1, 'CGPA': 2}
query_variable = 'Intelligence'
num_samples = 25  # You can adjust the number of samples

# Perform rejection sampling
samples = rejection_sampling(model, evidence, query_variable, num_samples)

print("with evidence a:1,cgpa:2",samples)

# Print the results
print(f"Rejection Sampling Results for P({query_variable} | {evidence}):")
print(f"Number of accepted samples: {len(samples)}")
print(f"Probability estimate: {sum(samples) /len(samples)}")


#Approximate Inference --Likelihood weighting

from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

cpd_intelligence = TabularCPD(variable= 'Intelligence', variable_card= 2 ,
                              values=[[ 0.7],[0.3]])

cpd_difficulty = TabularCPD( variable='Difficulty',variable_card= 2 ,
                             values=[[ 0.6], [0.4]])

cpd_cgpa = TabularCPD(variable= 'CGPA', variable_card= 3,
                      values=[[0.3,0.4,0.2,0.35 ],[0.3,0.2,0.3,0.45],[0.4,0.4,0.5,0.2]] ,
evidence=['Intelligence','Difficulty' ] ,evidence_card=[2,2 ])

cpd_gre = TabularCPD(variable= 'GRE', variable_card= 2 ,values= [[0.8,0.2] ,
[0.2, 0.8] ] , evidence=[ 'Intelligence' ] , evidence_card =[2])

cpd_admit = TabularCPD( variable='Admit', variable_card = 2 ,
values=[[ 0.6,0.2,0.55,0.6,0.6,0.3],[0.4,0.8,0.45 ,0.4,0.4,0.7] ],evidence= ['CGPA','GRE' ], evidence_card= [3, 2] )

model = BayesianNetwork([('Intelligence', 'CGPA'),
('Difficulty', 'CGPA'),
('CGPA', 'Admit'),
('Intelligence', 'GRE'),
('GRE', 'Admit')])

model.add_cpds(cpd_intelligence,cpd_difficulty,cpd_cgpa,
cpd_gre,cpd_admit)

assert model.check_model()

# Create the BayesianModelSampling object
sampling = BayesianModelSampling(model)

# Define evidence and query variable
evidence = {'Admit': 1,'CGPA': 1}
query_variable = 'Intelligence'
num_samples = 5
weights = []

# Rejection Sampling Function
def likelihood_weighting(model, evidence, query_variable, num_samples):

    counts = {value: 0 for value in model.get_cpds(query_variable).values}
    for key in counts:
        key_type = type(key)
        print(f"Key: {key}, Type: {key_type}")

    for _ in range(num_samples):
        
        sample = sampling.forward_sample(size=1).iloc[0]
        print(sample)
        print("------before checking up-----------")
        
        #checking sample based on evidence
        
        if all(sample[var] == value for var, value in evidence.items()):
        
            #sample=sample[query_variable]
            
            print("samples --->",sample)
            
            weight = 0.55
    
            for var, value in evidence.items():
                
                var_cpd = model.get_cpds(var)

                print("specific value from cpt for evidence",var)
                print(var_cpd.values[tuple(sample[var] for var in var_cpd.variables)])
        
                weight *= var_cpd.values[tuple(sample[var] for var in var_cpd.variables)]

                print("w1",weight)

                query_cpd = model.get_cpds(query_variable)
                
            ''' variables_only_in_sample = set(sample.keys()) - set(evidence.keys())
                
            for var in variables_only_in_sample:
                
                query_cpd = model.get_cpds(var)
                
                print("specific value from cpt for querry",var)
                print(query_cpd.values[tuple(sample[var] for var in query_cpd.variables)])
                
                weight *= query_cpd.values[tuple(sample[var] for var in query_cpd.variables)]

                print("w2",weight)'''
            
            weights.append(weight)
            counts[query_cpd.values[int(sample[query_variable])]] += 1
            
    return counts
        
counts=likelihood_weighting(model, evidence, query_variable, num_samples)

print("WEIGHTS :\n",weights)

if len(weights)>0:
    result=sum(weights)/len(weights)
    
    # Print the results
    print(f"Likelihood Weighting Results for P({query_variable} | {evidence}):")
    print(f"Counts for each value of {query_variable}: {counts}")
    print(f"Probability estimate: {round(result*100,2)}")
    
else:
    print("Increses the sample size the machine could not find the required sample")


