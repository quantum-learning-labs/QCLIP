import pickle, pandas as pd, numpy as np, sys
from scipy import stats
G='/cs/research/pplv/comp_bridge/qclip_snapshots/gap/'
R=pickle.load(open(G+'base_nodiscard.pkl','rb'))+pickle.load(open(G+'base_discard.pkl','rb'))
rows=[]
for r in R:
    df=r['df'].copy(); df['method']='discard' if r['discard'] else 'nodiscard'; df['seed']=r['seed']; df['split']=r['split']; df['pair']=np.arange(len(df)); rows.append(df)
D=pd.concat(rows)
def ms(x): return f"{x.mean():.3f}±{x.std(ddof=1):.3f}"
print('== accuracy by split (mean±sd over 10 seeds), several similarity functions')
for sp in ('train','val'):
    for sc in ('hs','fid','pauli_cos','nhs','trdist'):
        a=D[D.split==sp].groupby(['method','seed'])[sc+'_correct'].mean().unstack(0)
        print(f"{sp:5s} {sc:9s} nodiscard {ms(a.nodiscard)}  discard {ms(a.discard)}  gap {ms(a.nodiscard-a.discard)}")
V=D[D.split=='val'].copy()
print('len_match all:', V.len_match.all())
V['negtype']=np.select([V.subj_neg,V.verb_neg,V.obj_neg],['subj','verb','obj'],'other')
print('\n== val accuracy by negative type (hs)')
t=V.groupby(['negtype','method','seed']).hs_correct.mean().groupby(['negtype','method']).agg(['mean','std']).unstack(1).round(3)
t['n_pairs']=V[(V.method=='discard')&(V.seed==V.seed.iloc[0])].negtype.value_counts(); print(t)
print('\n== purity (val)')
print(V.groupby('method')[['pur_text','pur_pos','pur_neg','img_pair_hs','text_norm']].median().round(3))
d=V[V.method=='discard']
print('\n== discard: val accuracy by text purity quartile')
d=d.assign(pq=d.groupby('seed').pur_text.transform(lambda x: pd.qcut(x,4,labels=['Q1 most mixed','Q2','Q3','Q4 purest'])))
print(d.groupby('pq').agg(acc=('hs_correct','mean'),pur=('pur_text','median'),n_words=('n_words','mean')).round(3))
print('\n== accuracy by sentence length (n_words in diagram)')
print(V.groupby(['n_words','method']).hs_correct.mean().unstack().round(3).join(V[V.method=="discard"].groupby('n_words').size().rename('n')/10))
print('\n== discard purity vs n_words'); print(d.groupby('n_words').pur_text.median().round(3))
# per-pair agreement: are discard errors a superset of nodiscard errors?
pa=V.groupby(['method','pair']).hs_correct.mean().unstack(0)
print('\n== per-pair mean correctness over seeds; corr', round(pa.corr().iloc[0,1],3))
print('pairs nodiscard>=0.8 & discard<=0.2:', ((pa.nodiscard>=0.8)&(pa.discard<=0.2)).sum(), ' reverse:', ((pa.discard>=0.8)&(pa.nodiscard<=0.2)).sum(), ' of', len(pa))
# image pair similarity: hard pairs where the two images are close
V['imgq']=V.groupby(['method','seed']).img_pair_hs.transform(lambda x: pd.qcut(x,4,labels=['far','2','3','close']))
print('\n== val accuracy by how similar the two images are (quartile of Tr(rho_pos rho_neg))')
print(V.groupby(['imgq','method']).hs_correct.mean().unstack().round(3))
print(V.groupby('method').img_pair_hs.median().round(3))
