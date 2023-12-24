#Temporal_model --Filtering
import numpy as np

motion_model = np.array([[0.7,0.3],[0.3,0.7]])
observation_model = {"U":np.array([[0.9,0],[0,0.2]]),
                    "N":np.array([[0.1,0],[0,0.8]])}
prior = {0:np.array([[0.5],[0.5]])}
evidence = "UU"

print(f"DAY{0} : {prior[0]}")
for i in range(len(evidence)):
    prior_prob = np.matmul(motion_model,prior[i])
    post_prob = np.matmul(observation_model[evidence[i]],prior_prob)
    norm = post_prob/sum(post_prob)
    prior[i+1]=norm
    print(f"DAY{i+1} : {prior[i+1]}")


#Temporal Model -- Smoothing
import numpy as  np

mm = np.array([[0.7, 0.3],[0.3, 0.7]])
om = {"U" : np.array([[0.9, 0],[0, 0.2]]),
                     "N" : np.array([[0.1, 0],[0, 0.8]])}
pri = {0 : np.array([[0.5], [0.5]])}
evi = "UU"
k = 1

for i in range(len(evi)):
  pri_prob = np.matmul(mm, pri[i])
  post_prob = np.matmul(om[evi[i]],pri_prob)
  norm = post_prob/(sum(post_prob))
  pri[i+1]=norm

bwd = 1
for i in range(k, len(evi)):
  mat = np.dot(bwd, mm)
  bwd = np.dot(om[evi[i]], mat)

fwd = pri[k]
print("Before Smoothing:")
print(f"Rain in day {k}: {fwd}")

smoothing = np.dot(bwd, fwd)
smoothing = smoothing / sum(smoothing)
print("After Smoothing:")
print(f"Rain in day {k}: {smoothing}")