"""Measured synthetic calibration and same-support evidence, no corrected roof fiction."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from osgeo import gdal

P=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
OLD=P/'XIAN_R5_ALIGNMENT_FEATURE_RECOVERY_20261013'
OUT=P/'A_LOCAL_ALIGNMENT_NUMERICAL_AND_CHECK_CLOSURE_20261013'
FONT=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
    'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))


def title(ax,text):ax.set_title(text,fontproperties=FONT,loc='left',fontsize=12)


def save(fig,name,target=OUT):
    if any((target/(name+ext)).exists() for ext in ['.png','.pdf','_preview.png']):raise ValueError('Append-only figure exists')
    target.mkdir(exist_ok=True)
    fig.savefig(target/(name+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(target/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)
    im=Image.open(target/(name+'.png'));im.thumbnail((1500,1500));im.save(target/(name+'_preview.png'))


def synthetic():
    fixture=np.load(OUT/'OLD_FIXTURE_RECONSTRUCTED.npz');audit=read('SYNTHETIC_CALIBRATION_AUDIT.json')
    compare=read('AROSICS_NCC_COMPARISON.json');smooth=read('TEXTURE_SINGLE_FACTOR_PROBE.json')
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    vmin,vmax=np.percentile(fixture['reference'],[1,99])
    for ax,key,name in zip(axes[0,:2],['reference','moving'],['(a) 旧合成参考：Gaussian σ=1.2','(b) 已知内容位移 (+1.25,-0.75) px']):
        ax.imshow(fixture[key][64:192,64:192],cmap='gray',vmin=vmin,vmax=vmax,interpolation='nearest')
        title(ax,name);ax.set_xticks([]);ax.set_yticks([])
    ax=axes[0,2]
    for row,color in zip(audit['old_replay']['rows'],['#0072B2','#E69F00']):
        x,y=row['correction_m'];tx,ty=row['expected_correction_m']
        ax.quiver(0,0,tx,ty,angles='xy',scale_units='xy',scale=1,color=color,alpha=.35,width=.009)
        ax.quiver(0,0,x,y,angles='xy',scale_units='xy',scale=1,color=color,width=.006)
        ax.scatter(tx,ty,s=55,marker='x',color=color);ax.scatter(x,y,s=30,marker='o',color=color)
    ax.set_xlim(-16,16);ax.set_ylim(-12,12);ax.set_aspect('equal');ax.grid(alpha=.15)
    ax.set_xlabel('East (+x), metres');ax.set_ylabel('North (+y), metres')
    title(ax,'(c) 正反方向：× 真值 / ● 预测')
    ax.text(.02,.98,'Old forward error: 5.872m\nOld reverse error: 3.319m\nConvention error: 0m',
        transform=ax.transAxes,va='top',fontsize=10,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    ax=axes[1,0];cases=['ZERO','INTEGER','SUBPIXEL_X','SUBPIXEL_Y','BOTH'];x=np.arange(5)
    ar=[];ncc=[]
    for name in cases:
        ar.append(max(r['IMAGE_REGISTRATION_ESTIMATION_ERROR'] for r in compare['rows'] if r['case']==name and r['GSD']=='SAME_10M' and r['algorithm']=='AROSICS_COREG'))
        ncc.append(max(r['IMAGE_REGISTRATION_ESTIMATION_ERROR'] for r in compare['rows'] if r['case']==name and r['GSD']=='SAME_10M' and r['algorithm']=='NCC_EXISTING_INTEGER'))
    ax.bar(x-.18,ar,.35,color='#0072B2',label='AROSICS');ax.bar(x+.18,ncc,.35,color='#E69F00',label='NCC integer')
    ax.axhline(2,color='#D55E00',ls='--',lw=1,label='2m gate');ax.set_xticks(x,['Zero','Integer','Sub-x','Sub-y','Both'],rotation=20)
    ax.set_ylabel('Max forward/reverse error (m)');ax.legend(fontsize=9)
    title(ax,'(d) 宽频 v2 对照，不替换旧失败')
    ax=axes[1,1];sig=[r['smoothing_sigma_px'] for r in smooth['rows']];err=[r['error_m'] for r in smooth['rows']]
    ax.plot(sig,err,'o-',color='#0072B2');ax.axhline(2,color='#D55E00',ls='--',lw=1)
    ax.set_xlabel('Only varied factor: Gaussian sigma (px)');ax.set_ylabel('AROSICS estimation error (m)')
    for s,e in zip(sig,err):ax.annotate(f'{e:.2f}',(s,e),xytext=(0,7),textcoords='offset points',ha='center')
    title(ax,'(e) 同随机场：低通纹理敏感性')
    ax=axes[1,2];ax.axis('off')
    ax.text(0,1,'SYNTHETIC NUMERICAL GATE = FAIL\n\nOld failure reproduced exactly.\n2m threshold unchanged.\nBroadband controls pass (<0.864m),\nbut Gaussian1.2 stress case still fails.\n\nCross-GSD control errors:\nforward 0.097m; reverse 0.045m.\nNot evidence of real ground accuracy.\n\nNo real displacement applied.',va='top',fontsize=11)
    fig.suptitle('SYNTHETIC CALIBRATION — CONVENTION ERROR vs ESTIMATION ERROR',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.95));save(fig,'AROSICS_SYNTHETIC_CALIBRATION')


def real():
    plan=read('FIT_CHECK_REGION_DESIGN.json');diag=read('READ_ONLY_REAL_DIAGNOSTIC.json')
    ds=gdal.Open(str(OLD/'GE_NATIVE_CONTEXT_DISPLAY_1M.tif'))
    ge=ds.ReadAsArray(1920,1280,640,640).transpose(1,2,0);ds=None
    z=np.load(OLD/'MASK_DIAGNOSTIC_DATA.npz');ref=z['ref'][128:192,192:256]
    fig,axes=plt.subplots(1,3,figsize=(15,6))
    axes[0].imshow(ge,extent=(192,256,192,128),interpolation='nearest');title(axes[0],'原始 GE：同一冻结区域 roi_r2_c3')
    axes[1].imshow(ref,extent=(192,256,192,128),interpolation='nearest');title(axes[1],'原生10m参考：没有新增定位信息')
    for f,color,marker in zip(plan['rows'],['#0072B2','#E69F00'],['s','^']):
        c,r=f['anchor_col_row'];label='FIT roundabout' if f['fold']=='FIT' else 'CHECK north junction'
        for ax in axes[:2]:
            ax.scatter(c+.5,r+.5,s=100,marker=marker,facecolors='none',edgecolors=color,lw=2,label=label)
            ax.text(c+1,r-1,f['fold']+' PATCH',color='white',fontsize=10,bbox={'facecolor':'black','alpha':.8,'edgecolor':'none','pad':2})
        row=next(x for x in diag['rows'] if x['feature_id']==f['feature_id'] and x['mask_policy']=='A1')
        vec=row.get('numeric_candidate_vector_m')
        if vec:
            axes[1].annotate('',xy=(c+.5+vec[0]/10,r+.5-vec[1]/10),xytext=(c+.5,r+.5),
                arrowprops={'arrowstyle':'->','color':color,'lw':1.5})
    for ax in axes[:2]:
        ax.set_xlim(192,256);ax.set_ylim(192,128);ax.set_xlabel('Original target-grid column');ax.set_ylabel('Original target-grid row')
    axes[1].set_xticks(np.arange(192,257,8));axes[1].set_yticks(np.arange(128,193,8));axes[1].grid(alpha=.15)
    axes[2].axis('off');title(axes[2],'未接受原因：不是纠偏后效果')
    text='SAME REGION PATCH DESIGN: 1 FIT + 1 CHECK\nFORMAL REQUIREMENT: 3 FIT + 3 CHECK\nPRECISION ELIGIBLE TIES: 0\n\nSYNTHETIC NUMERICAL GATE: FAIL\nA2 verified ground mask: NOT AVAILABLE\nReference localization: UNRESOLVED\nFormal residual / uncertainty: null\n\n'
    for f in plan['rows']:
        row=next(x for x in diag['rows'] if x['feature_id']==f['feature_id'] and x['mask_policy']=='A1')
        text+=f'{f["fold"]} A1 numeric candidate:\n  {row["numeric_candidate_vector_m"]} m\n'
    text+='\nNO CORRECTION / NO ROOF MOVEMENT\nABSOLUTE ACCURACY = UNKNOWN'
    axes[2].text(0,.97,text,va='top',fontsize=10)
    fig.suptitle('LOCAL ALIGNMENT EVIDENCE — READ-ONLY, SAME SUPPORT, ZERO APPLICATION',fontsize=15)
    fig.text(.02,.02,'Anchors and arrows are candidates, not ties. GE actual date 2018-11-01; reference 2019 exact window unknown. All16 denominator preserved.',fontsize=10)
    fig.tight_layout(rect=(0,.14,1,.92));save(fig,'LOCAL_ALIGNMENT_EVIDENCE',OUT/'FIGURE_FIXED_02')


if __name__=='__main__':
    if not (OUT/'AROSICS_SYNTHETIC_CALIBRATION.png').exists():synthetic()
    real()
