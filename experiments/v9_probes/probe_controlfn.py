# Probe: control-function (selection-corrected) item appeal on KuaiRand 10K. Item effects a_i are fitted on
# exposed pairs together with a covariate h(e_ui), e_ui = personalised exposure score learned from impressions.
import sys, numpy as np, torch
sys.path.insert(0, sys.argv[1])
from harness import load, std, Selector, report
from drup.metrics import evaluate
torch.set_num_threads(4)
D=load('kuairand'); O,Y=D['O'],D['Y']; rows=D['rows']; ro=D['row_of']; m,n=O.shape
idx=O.nonzero(as_tuple=True); u_,i_=idx; y=Y[idx].double()
def ev(S): return round(float(np.mean([evaluate(S, t, D['ks'], ro)[D['key']] for _,t in D['splits']])),4)
def fit(cov, lam=5.0, it=30):
    # y = mu + b_u + a_i + sum_k rho_k cov_k  (alternating ridge / least squares)
    mu=y.mean(); bu=torch.zeros(m,dtype=torch.float64); ai=torch.zeros(n,dtype=torch.float64)
    rho=torch.zeros(len(cov),dtype=torch.float64); C=torch.stack(cov,1) if cov else None
    cu=torch.bincount(u_,minlength=m).double(); ci=torch.bincount(i_,minlength=n).double()
    for _ in range(it):
        r0=y-mu-(C@rho if cov else 0)
        bu=torch.zeros(m,dtype=torch.float64).index_add_(0,u_,r0-ai[i_])/(cu+lam)
        ai=torch.zeros(n,dtype=torch.float64).index_add_(0,i_,r0-bu[u_])/(ci+lam)
        if cov:
            r=y-mu-bu[u_]-ai[i_]; rho=torch.linalg.lstsq(C,r[:,None]).solution[:,0]
    return ai, rho
X=O.clone()
G=X.T@X
for lam in (3000.,):
    B=torch.linalg.inv(G+lam*torch.eye(n)); B=-B/torch.diag(B)[None,:]; B.fill_diagonal_(0)
E=(O@B)  # exposure score from impression co-occurrence (all users)
e=E[idx].double()
ez=(e-e.mean())/e.std()
pop=torch.log1p(O.sum(0)).double(); popz=(pop-pop.mean())/pop.std()
ai0,_=fit([]); print('additive (no covariate)', ev(ai0.float()[None,:].expand(len(rows),-1)))
for name,cov in [('e',[ez]),('e,e^2',[ez,ez**2]),('logpop_i',[popz[i_]]),('e,logpop',[ez,popz[i_]])]:
    ai,rho=fit(cov); print('control', name, 'rho', rho.numpy().round(4), ev(ai.float()[None,:].expand(len(rows),-1)))
