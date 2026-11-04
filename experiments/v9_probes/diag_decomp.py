# Is the gain of a personal log model over popularity personal or item-level? (diagnostic, KuaiRand 10K)
import sys, numpy as np, torch
sys.path.insert(0, sys.argv[1])
from harness import load, std
from drup.metrics import evaluate
out=open(sys.argv[2],'a')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); out.write(s+'\n'); out.flush()
torch.set_num_threads(4)
D=load('kuairand'); O,Y=D['O'],D['Y']; rows=D['rows']; ro=D['row_of']
X=(O*Y)
def ev(S):  # test of all 5 splits, mean
    return round(float(np.mean([evaluate(S, t, D['ks'], ro)[D['key']] for _,t in D['splits']])),4)
pop=X.sum(0); P('decomp pop (clicks)', ev(pop[None,:].expand(len(rows),-1)))
P('decomp pop (impressions)', ev(O.sum(0)[None,:].expand(len(rows),-1)))
G=X.T@X
for lam in (300.,3000.,30000.):
    B=torch.linalg.inv(G+lam*torch.eye(G.shape[0])); B=-B/torch.diag(B)[None,:]; B.fill_diagonal_(0)
    S=X[rows]@B
    Sall=X@B
    item=Sall.mean(0)
    P(f'decomp EASE lam={lam}: personal', ev(S), 'item-avg of EASE', ev(item[None,:].expand(len(rows),-1)), 'personal - user mean + item avg(std)',
      ev(std(S-S.mean(1,keepdim=True))+0*item))
    for g in (0.25,1.0):
        P('   pop-std + g*personal-std', g, ev(std(torch.log1p(pop))[None,:]+g*std(S-S.mean(1,keepdim=True))))
