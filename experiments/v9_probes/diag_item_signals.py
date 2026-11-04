import pandas as pd, numpy as np, sys
d=sys.argv[1]; out=open(sys.argv[2],'a')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); out.write(s+'\n'); out.flush()
rnd=pd.read_csv(d+'/log_random_4_22_to_5_08_pure.csv')
std=pd.concat([pd.read_csv(d+'/log_standard_4_08_to_4_21_pure.csv'),pd.read_csv(d+'/log_standard_4_22_to_5_08_pure.csv')])
st=pd.read_csv(d+'/video_features_statistic_pure.csv'); vb=pd.read_csv(d+'/video_features_basic_pure.csv')
P('stat cols',list(st.columns)); P('stat rows',len(st),'unique vids',st.video_id.nunique())
ur=rnd.groupby('video_id').agg(n=('is_click','size'),c=('is_click','mean'),lv=('long_view','mean'))
ur=ur[ur.n>=50]
std['pr']=std.play_time_ms/std.duration_ms.clip(lower=1)
def rate(x,col,lam=20):
    g=x.groupby('video_id')[col].agg(['sum','count']); mu=x[col].mean(); return (g['sum']+lam*mu)/(g['count']+lam)
feats={'log ctr all tabs':rate(std,'is_click'),'log ctr tab1':rate(std[std.tab==1],'is_click'),
 'log longview':rate(std,'long_view'),'log longview tab1':rate(std[std.tab==1],'long_view'),'log like':rate(std,'is_like'),
 'log playratio':rate(std,'pr'),'log impressions (pop)':std.groupby('video_id').size(),'log clicks (pop)':std.groupby('video_id').is_click.sum()}
s2=st.set_index('video_id')
for c in ['show_cnt','play_cnt','complete_play_cnt','valid_play_cnt','long_time_play_cnt','like_cnt','follow_cnt','share_cnt']:
    if c in s2: feats['platform '+c+'/show']=s2[c]/s2.show_cnt.clip(lower=1)
feats['platform show_cnt']=s2.show_cnt
for k,f in feats.items():
    j=ur.join(f.rename('f'),how='inner')
    P(f'{k:35s} n={len(j)} spearman(click)={j[["c","f"]].corr("spearman").iloc[0,1]:.3f} spearman(longview)={j[["lv","f"]].corr("spearman").iloc[0,1]:.3f}')
P('tab x click std', std.groupby('tab').is_click.mean().round(3).to_dict())
P('date range std',std.date.min(),std.date.max(),'rnd',rnd.date.min(),rnd.date.max())
P('upload dt range', vb.upload_dt.min(), vb.upload_dt.max())
