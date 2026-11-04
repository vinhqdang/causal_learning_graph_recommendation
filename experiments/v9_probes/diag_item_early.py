# Item level: is the early-life (exploration-phase) response of a new video a better appeal estimate?
import pandas as pd, numpy as np, sys
d=sys.argv[1]; out=open(sys.argv[2],'a')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); out.write(s+'\n'); out.flush()
rnd=pd.read_csv(d+'/log_random_4_22_to_5_08_pure.csv',usecols=['video_id','is_click','long_view'])
std=pd.concat([pd.read_csv(d+'/log_standard_4_08_to_4_21_pure.csv'),pd.read_csv(d+'/log_standard_4_22_to_5_08_pure.csv')]).sort_values('time_ms')
ur=rnd.groupby('video_id').agg(n=('is_click','size'),c=('is_click','mean')); ur=ur[ur.n>=50]
std['k']=std.groupby('video_id').cumcount()
mu=std.is_click.mean()
def sp(f): j=ur.join(f.rename('f'),how='inner'); return round(j[['c','f']].corr('spearman').iloc[0,1],3), len(j)
for K in (10,20,50,100,10**9):
    x=std[std.k<K]; g=x.groupby('video_id').is_click.agg(['sum','count'])
    P(f'early CTR first {K} impressions: spearman', sp((g['sum']+10*mu)/(g['count']+10)), ' clicks count', sp(g['sum']))
for t in sorted(std.tab.unique()):
    x=std[std.tab==t]
    if len(x)<20000: continue
    g=x.groupby('video_id').is_click.agg(['sum','count'])
    P(f'tab {t} ({len(x)} imps): CTR spearman', sp((g['sum']+10*x.is_click.mean())/(g['count']+10)), 'clicks', sp(g['sum']), 'imps', sp(g['count']))
# users with low activity (less targeted?) 
ua=std.groupby('user_id').size()
for lo,hi in [(0,20),(20,60),(60,10**9)]:
    x=std[std.user_id.map(ua).between(lo,hi-1)]; g=x.groupby('video_id').is_click.agg(['sum','count'])
    P(f'users with {lo}-{hi} imps: CTR', sp((g['sum']+10*x.is_click.mean())/(g['count']+10)), 'clicks', sp(g['sum']))
