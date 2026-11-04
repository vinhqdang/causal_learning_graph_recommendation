# Diagnostic (item level): do temporal exposure dynamics of the standard log predict the uniform click rate?
import pandas as pd, numpy as np, sys
d=sys.argv[1]; out=open(sys.argv[2],'a')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); out.write(s+'\n'); out.flush()
rnd=pd.read_csv(d+'/log_random_4_22_to_5_08_pure.csv',usecols=['video_id','is_click','long_view'])
std=pd.concat([pd.read_csv(d+'/log_standard_4_08_to_4_21_pure.csv'),pd.read_csv(d+'/log_standard_4_22_to_5_08_pure.csv')])
vb=pd.read_csv(d+'/video_features_basic_pure.csv').set_index('video_id')
ur=rnd.groupby('video_id').agg(n=('is_click','size'),c=('is_click','mean'))
ur=ur[ur.n>=50]
std['day']=pd.to_datetime(std.date.astype(str)).dt.dayofyear
g=std.groupby('video_id')
f=pd.DataFrame({'imp':g.size(),'clk':g.is_click.sum(),'lv':g.long_view.sum(),
 'imp_early':std[std.date<20220422].groupby('video_id').size(),
 'imp_late':std[std.date>=20220422].groupby('video_id').size(),
 'clk_late':std[std.date>=20220422].groupby('video_id').is_click.sum(),
 'users':g.user_id.nunique(),'tab1_share':g.tab.apply(lambda t:(t==1).mean()),
 'ctr':g.is_click.mean(),'last_day':g.day.max(),'first_day':g.day.min()}).fillna(0)
f['trend']=np.log1p(f.imp_late)-np.log1p(f.imp_early)
f['dur']=vb.video_duration
for c in f.columns:
    j=ur.join(f[c].rename('f'),how='inner')
    P(f'{c:12s} spearman with uniform click rate {j[["c","f"]].corr("spearman").iloc[0,1]:+.3f}')
