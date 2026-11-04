# Within-user temporal context diagnostic on KuaiRand-Pure: does the user's state just before a random
# impression (from the standard log only, strictly earlier timestamps) rank the random impressions?
import pandas as pd, numpy as np, sys
d=sys.argv[1]; out=open(sys.argv[2],'a')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); out.write(s+'\n'); out.flush()
rnd=pd.read_csv(d+'/log_random_4_22_to_5_08_pure.csv')
std=pd.concat([pd.read_csv(d+'/log_standard_4_08_to_4_21_pure.csv'),pd.read_csv(d+'/log_standard_4_22_to_5_08_pure.csv')])
vb=pd.read_csv(d+'/video_features_basic_pure.csv').set_index('video_id')
rnd=rnd.sort_values('time_ms'); std=std.sort_values('time_ms')
# item LOO oracle and pop
si=rnd.groupby('video_id').is_click.transform('sum'); ci=rnd.groupby('video_id').is_click.transform('size')
rnd['loo']=(si-rnd.is_click)/(ci-1).clip(lower=1)
rnd['pop']=rnd.video_id.map(std.groupby('video_id').is_click.sum()).fillna(0)
rnd['dur']=-rnd.video_id.map(vb.video_duration).fillna(1e5)
# previous std impressions of the same user (merge_asof backward, strictly before)
std['one']=1
g=std.groupby('user_id')
std['cumclk']=g.is_click.cumsum(); std['cumn']=g.one.cumsum()
std['cumlv']=g.long_view.cumsum()
m=pd.merge_asof(rnd[['user_id','time_ms']].reset_index(),std[['user_id','time_ms','cumclk','cumn','cumlv']].rename(columns={'time_ms':'t_prev'}),left_on='time_ms',right_on='t_prev',by='user_id',allow_exact_matches=False,direction='backward').set_index('index')
rnd['gap']=(rnd.time_ms-m.t_prev)/60000.0
# recent click rate: last 20 std impressions before t (difference of cumulative counts)
std_idx=std.set_index(['user_id']).sort_values('time_ms')
cum=m.cumclk.fillna(0).values; cn=m.cumn.fillna(0).values
# approximate window: rate over all earlier impressions vs. global user rate
urate=std.groupby('user_id').is_click.mean()
rnd['state_all']=np.where(cn>0,cum/np.maximum(cn,1),np.nan)
rnd['state_minus_user']=rnd.state_all-rnd.user_id.map(urate)
rnd['recent']=-rnd.gap  # more recent std activity -> active session
rnd['hour']=rnd.hourmin//100
def ev(col, df=None):
    df=rnd if df is None else df
    vals=[]
    for u,gg in df.groupby('user_id'):
        r=gg.is_click.values
        if r.sum()==0: continue
        s=gg[col].fillna(gg[col].mean() if gg[col].notna().any() else 0).values
        o=np.argsort(-s,kind='stable'); top=r[o][:10]; disc=1/np.log2(np.arange(2,len(top)+2))
        vals.append((top*disc).sum()/disc[:int(min(r.sum(),10))].sum())
    return round(float(np.mean(vals)),4), len(vals)
rnd['rand']=np.random.default_rng(0).random(len(rnd))
for c in ['rand','pop','dur','loo','state_all','recent']:
    P('ctx-diag rank by',c,ev(c))
rnd['loo_plus_recent']=rnd.loo.rank(pct=True)+0.5*(rnd.groupby('user_id').recent.rank(pct=True))
P('ctx-diag rank by loo+0.5*recent-rank',ev('loo_plus_recent'))
# click rate by gap bucket
b=pd.cut(rnd.gap,[0,1,5,30,120,1440,1e9])
P('click rate by minutes since previous std impression', rnd.groupby(b).is_click.agg(['mean','size']).round(3).to_dict())
P('click rate by hour', rnd.groupby('hour').is_click.mean().round(3).to_dict())
