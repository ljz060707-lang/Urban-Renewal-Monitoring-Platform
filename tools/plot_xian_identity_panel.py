"""Render actual GE pixels and original polygons; no synthetic imagery."""
import json
from pathlib import Path
import textwrap
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from PIL import Image

OUT=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/23_building_object_identity/XIAN_R5_IDENTITY_MINIMAL_CLOSURE_20261015')
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'font.size':10,'axes.unicode_minus':False,'pdf.fonttype':42})
revised='--revision' in sys.argv
cases=json.loads((OUT/('PANEL_CASES_FIXED_02.json' if revised else 'PANEL_CASES.json')).read_text(encoding='utf-8'))['cases']
rgb=json.loads((OUT/('PANEL_RGB_PROVENANCE_FIXED_02.json' if revised else 'PANEL_RGB_PROVENANCE.json')).read_text(encoding='utf-8'))['records']
assets=OUT/('panel_assets_fixed_02' if revised else 'panel_assets')
destination=OUT/'figure_fixed_02' if revised else OUT
if revised:destination.mkdir(exist_ok=False)
fig,axes=plt.subplots(len(cases),2,figsize=(11,4.7*len(cases)),squeeze=False)
colors={'roof':'#56B4E9','object':'#F0E442','gt':'#CC79A7'}

def draw(ax,geom,color,style,fill):
    polygons=[geom['coordinates']] if geom['type']=='Polygon' else geom['coordinates']
    for polygon in polygons:
        for j,ring in enumerate(polygon):
            ax.add_patch(Polygon(ring,closed=True,facecolor=color if fill and j==0 else 'none',
                alpha=.15 if fill and j==0 else 1,edgecolor='none' if fill else color,
                linewidth=1.4,linestyle=style))
            if fill:ax.add_patch(Polygon(ring,closed=True,facecolor='none',edgecolor=color,linewidth=1.8))

for i,(c,p) in enumerate(zip(cases,rgb)):
    x0,y0,x1,y1=p['display_bounds'];image=Image.open(assets/f'case_{i+1}_GE_DISPLAY.png')
    for j,ax in enumerate(axes[i]):
        ax.imshow(image,extent=(x0,x1,y0,y1),interpolation='nearest');ax.set_xlim(x0,x1);ax.set_ylim(y0,y1)
        ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
        sx=x0+.07*(x1-x0);sy=y0+.07*(y1-y0);length=20 if x1-x0<160 else 40
        ax.plot([sx,sx+length],[sy,sy],color='white',linewidth=3)
        ax.text(sx+length/2,sy+.025*(y1-y0),f'{length} m',ha='center',color='white',fontsize=9)
        ax.text(.97,.97,'N ↑',transform=ax.transAxes,ha='right',va='top',color='white',fontsize=10)
    axes[i,0].set_title(f'{i+1}. {c["title"]}｜原始 GE RGB\n{c["observation_id"]}',loc='left',fontsize=11,pad=8)
    ax=axes[i,1]
    draw(ax,c['observed_geometry'],colors['roof'],'-',True)
    for uid,g in c['object_geometries'].items():draw(ax,g,colors['object'],'--',False)
    for fid,g in c['manual_geometries'].items():draw(ax,g,colors['gt'],':',False)
    ids=', '.join(c['building_object_ids']) or '无可用对象关联'
    caption='\n'.join(textwrap.wrap(ids,width=58))
    gtids=', '.join(c['manual_reference_fids']) or '无候选参考'
    gtcaption='\n'.join(textwrap.wrap('GT 原始 FID: '+gtids,width=62))
    ax.set_title(f'原始轮廓叠加｜{c["result"]}\n{caption}\n{gtcaption}',loc='left',fontsize=8,pad=8)
    ax.text(.01,-.045,f'证据：{c["evidence_level"]}；physical_id = NULL',transform=ax.transAxes,fontsize=9,va='top')
legend=[Line2D([0],[0],color=colors['roof'],lw=2,label='observed roof（青色半透明）'),
        Line2D([0],[0],color=colors['object'],lw=2,ls='--',label='既有 building object（黄虚线）'),
        Line2D([0],[0],color=colors['gt'],lw=2,ls=':',label='2019 人工屋顶参考（紫点线）')]
fig.suptitle('XIAN-R5-I｜单期屋顶—既有对象来源证据',fontsize=17,y=.994)
fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.973),ncol=3,frameon=False,fontsize=9)
fig.text(.5,.016,'GE 实际日期：2018-11-01；人工参考年份：2019。轮廓未平移。单期来源关联不证明跨期物理身份或屋顶纯度。\n所有案例均为 development；无独立验证；绝对位置精度 UNKNOWN。',ha='center',fontsize=10)
fig.subplots_adjust(top=.91,bottom=.055,left=.035,right=.98,hspace=.52,wspace=.09)
for suffix in ['png','pdf']:
    path=destination/f'IDENTITY_LINK_EVIDENCE_PANEL.{suffix}'
    if path.exists():raise FileExistsError(path)
    fig.savefig(path,dpi=300,facecolor='white')
plt.close(fig)
print(str(OUT/'IDENTITY_LINK_EVIDENCE_PANEL.png'))
