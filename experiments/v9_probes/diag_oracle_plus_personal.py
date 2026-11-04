# Diagnostic: leave-one-user-out uniform item rate (oracle, never a method) combined with a log personal model.
import sys, numpy as np, torch
sys.path.insert(0, sys.argv[1])
from harness import load, std
from drup.metrics import evaluate
out=open(sys.argv[2],'a')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); out.write(s+'\n'); out.flush()
torch.set_num_threads(4)
D=load('kuairand'); O,Y=D['O'],D['Y']; rows=D['rows']; ro=D['row_of']; n=O.shape[1]
test=[t for t in D['splits'][0][0]+D['splits'][0][1]]
si=np.zeros(n); ci=np.zeros(n)
for u,it,r in test: np.add.at(si,it,r); np.add.at(ci,it,1)
# LOO matrix: rows x n where for user u, item i: (si - r_ui)/(ci-1)
base=torch.tensor(si/np.maximum(ci,1)).float()
L=base[None,:].repeat(len(rows),1)
for u,it,r in test:
    L[ro[u],it]=torch.tensor((si[it]-r)/np.maximum(ci[it]-1,1)).float()
def ev(S): return round(float(np.mean([evaluate(S, t, D['ks'], ro)[D['key']] for _,t in D['splits']])),4)
P('oracle LOO item rate (10K users)', ev(L))
X=O*Y; G=X.T@X; B=torch.linalg.inv(G+3000*torch.eye(n)); B=-B/torch.diag(B)[None,:]; B.fill_diagonal_(0)
S=X[rows]@B; Sp=std(S-S.mean(1,keepdim=True))
Ls=std(torch.logit(L.clamp(1e-3,1-1e-3)))
for g in (0.1,0.25,0.5,1.0): P('oracle + g*EASE personal', g, ev(Ls+g*Sp))
