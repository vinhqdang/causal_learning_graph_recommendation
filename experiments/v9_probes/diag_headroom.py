# Headroom diagnostics on KuaiRand-Pure (all users) for several labels.
import pandas as pd, numpy as np, sys, scipy.sparse as sp
from scipy.sparse.linalg import svds
d=sys.argv[1]; out=open(sys.argv[2],'a')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); out.write(s+'\n'); out.flush()
rnd=pd.read_csv(d+'/log_random_4_22_to_5_08_pure.csv')
std=pd.concat([pd.read_csv(d+'/log_standard_4_08_to_4_21_pure.csv'),pd.read_csv(d+'/log_standard_4_22_to_5_08_pure.csv')])
users=np.union1d(rnd.user_id.unique(),std.user_id.unique()); items=np.union1d(rnd.video_id.unique(),std.video_id.unique())
m,n=len(users),len(items)
for x in (rnd,std): x['u']=np.searchsorted(users,x.user_id); x['i']=np.searchsorted(items,x.video_id)
def ndcg(S, R, mask, k=10):
    # S,R,mask dense (U,L)
    res=[]
    for s,r,mk in zip(S,R,mask):
        s=s[mk]; r=r[mk]
        if r.sum()==0: continue
        o=np.argsort(-s,kind='stable'); top=r[o][:k]; disc=1/np.log2(np.arange(2,len(top)+2))
        res.append((top*disc).sum()/disc[:int(min(r.sum(),k))].sum())
    return np.mean(res), len(res)
rng=np.random.default_rng(0)
stdkey=set(zip(std.u.values,std.i.values))
rnd=rnd.groupby(['u','i']).agg({c:'max' for c in ['is_click','long_view','is_like']}).reset_index()
rnd=rnd[[ (a,b) not in stdkey for a,b in zip(rnd.u.values,rnd.i.values)]]
rnd['A']=rng.random(len(rnd))<0.5
for lab in ['is_click','long_view']:
    sg=std.groupby(['u','i'])[lab].max().reset_index()
    # additive prior no-IPS (lam 5)
    y=sg[lab].values.astype(float); mu=y.mean(); bu=np.zeros(m); bi=np.zeros(n)
    for _ in range(20):
        bu=np.bincount(sg.u,y-mu-bi[sg.i],m)/(np.bincount(sg.u,None,m)+5)
        bi=np.bincount(sg.i,y-mu-bu[sg.u],n)/(np.bincount(sg.i,None,n)+5)
    prior_i=bi  # within-user ranking only depends on item part
    # LOO item oracle on random: (sum_i - y_ui)/(cnt_i -1)
    yr=rnd[lab].values.astype(float); si=np.bincount(rnd.i,yr,n); ci=np.bincount(rnd.i,None,n)
    loo=(si[rnd.i]-yr)/np.maximum(ci[rnd.i]-1,1)
    def evalcol(score, sel):
        df=pd.DataFrame({'u':rnd.u.values[sel],'s':score[sel] if len(score)==len(rnd) else score,'r':yr[sel]})
        vals=[]
        for u,g in df.groupby('u'):
            r=g.r.values
            if r.sum()==0: continue
            o=np.argsort(-g.s.values,kind='stable'); top=r[o][:10]; disc=1/np.log2(np.arange(2,len(top)+2))
            vals.append((top*disc).sum()/disc[:int(min(r.sum(),10))].sum())
        return np.mean(vals), len(vals)
    allsel=np.ones(len(rnd),bool)
    P(lab,'rand rate',yr.mean().round(4),'H1 all users: prior',evalcol(prior_i[rnd.i.values],allsel),'LOO item oracle',evalcol(loo,allsel),'random',evalcol(rng.random(len(rnd)),allsel))
    # H2: fit on A, evaluate B
    A=rnd.A.values; B=~A
    sa=np.bincount(rnd.i[A],yr[A],n); ca=np.bincount(rnd.i[A],None,n); mu_a=yr[A].mean()
    im=(sa+5*mu_a)/(ca+5)
    ua=np.bincount(rnd.u[A],yr[A]-im[rnd.i[A]],m)/(np.bincount(rnd.u[A],None,m)+5)
    res=yr[A]-im[rnd.i.values[A]]-ua[rnd.u.values[A]]
    M=sp.csr_matrix((res,(rnd.u.values[A],rnd.i.values[A])),shape=(m,n))
    P(lab,'H2 B: item-only(A)',evalcol(im[rnd.i.values],B),'log prior',evalcol(prior_i[rnd.i.values],B))
    for k in (4,16,64):
        U,s,Vt=svds(M,k=k)
        pers=np.einsum('ij,ij->i',U[rnd.u.values]*s,Vt.T[rnd.i.values])
        r=[]
        for c in (0.5,1,2,4):
            r.append((c,evalcol(im[rnd.i.values]+c*pers,B)[0].round(4)))
        P(lab,'H2 rank',k,r)
