import pandas as pd, numpy as np, sys
d=sys.argv[1]
rnd=pd.read_csv(d+'/log_random_4_22_to_5_08_pure.csv')
std=pd.concat([pd.read_csv(d+'/log_standard_4_08_to_4_21_pure.csv'),pd.read_csv(d+'/log_standard_4_22_to_5_08_pure.csv')])
for name,x in [('rnd',rnd),('std',std)]:
    print(name,len(x),x.user_id.nunique(),x.video_id.nunique(), 'is_rand', x.is_rand.mean())
    print(x[['is_click','is_like','is_follow','is_comment','is_forward','is_hate','long_view','is_profile_enter']].mean().round(4).to_dict())
    print('tab',x.tab.value_counts(normalize=True).round(3).to_dict())
    print('per user', x.groupby('user_id').size().describe().round(1).to_dict())
    print('dup pairs', x.duplicated(['user_id','video_id']).mean())
print('rand items in std', np.isin(rnd.video_id.unique(), std.video_id.unique()).mean())
k=set(zip(std.user_id,std.video_id)); ov=np.array([(u,i) in k for u,i in zip(rnd.user_id,rnd.video_id)]); print('rand pairs also in std',ov.mean())
print('rand users in std', np.isin(rnd.user_id.unique(), std.user_id.unique()).mean())
print(pd.crosstab(rnd.tab, rnd.is_click, normalize='index').round(3))
print('click|longview corr', rnd[['is_click','long_view','is_like']].corr().round(3))
