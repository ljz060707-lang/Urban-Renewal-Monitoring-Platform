from ml import *
model=Net(3,3).cuda().eval();model.load_state_dict(torch.load(OUT/'facade_best.pt',map_location='cpu')['model']);dl=loader(FacadeData('TUNE'),8)
bins=1000;hp=np.zeros(bins,'int64');hn=np.zeros(bins,'int64');total_positive=0
with torch.inference_mode():
 for x,y in dl:
  z,_=model(x.cuda(non_blocking=True));p=z.softmax(1).cpu().numpy();lab=y.numpy();valid=lab!=255;keep=valid&(p[:,1]<.2);b=np.minimum((p[:,2]*bins).astype('int64'),bins-1);positive=lab==2;total_positive+=int((positive&valid).sum());hp+=np.bincount(b[keep&positive],minlength=bins);hn+=np.bincount(b[keep&~positive],minlength=bins)
tp=np.cumsum(hp[::-1])[::-1];fp=np.cumsum(hn[::-1])[::-1];rows=[]
for i in range(500,bins):
 precision=float(tp[i]/max(tp[i]+fp[i],1));recall=float(tp[i]/max(total_positive,1));rows.append(dict(threshold=i/bins,precision=precision,recall=recall,true_positive_pixels=int(tp[i]),false_positive_pixels=int(fp[i])))
df=pd.DataFrame(rows);df.to_csv(OUT/'facade_threshold_calibration.csv',index=False);eligible=df[(df.precision>=.90)&(df.recall>=.01)&(df.true_positive_pixels>=100)]
if len(eligible):
 best=eligible.sort_values(['recall','precision'],ascending=False).iloc[0].to_dict();admitted=True
else:best=df.sort_values(['precision','recall'],ascending=False).iloc[0].to_dict();admitted=False
save('facade_threshold_parameters.json',dict(admitted=admitted,**best,minimum_source_precision=.90,minimum_source_recall=.01,roof_probability_cap=.20,calibration_role='BANDON_TUNE_SAME_POOL_AS_SELECTION_NOT_BLIND_NOT_TARGET_DOMAIN',pixel_semantics='BANDON projected geometric facade auxiliary',additional_target_rule='AND legacy E3 source-image risk; no unlabelled GT treated as background'))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots();ax.plot(df.threshold,df.precision,label='TUNE precision');ax.plot(df.threshold,df.recall,label='TUNE recall');ax.axhline(.9,ls='--',color='gray');ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(OUT/'facade_threshold_calibration.png');plt.close(fig)
status('FACADE_THRESHOLD_CALIBRATED',admitted=admitted,**best)
