import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
import seaborn as sns
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":150,"savefig.bbox":"tight",
  "axes.titleweight":"bold","axes.titlesize":14,"axes.labelsize":12,
  "legend.fontsize":9.5,"xtick.labelsize":10,"ytick.labelsize":10,"font.family":"DejaVu Sans"})
d=pd.read_csv("combined.csv")
LAT="latency_median_ms_max_rank"
pp_order=["tp1-ep1","tp1-ep2","tp1-ep4","tp2-ep2"]
B=d[d.study_name=="routing-sweep-qwen3"].copy()
rmodes=["uniform","random","zipfian","skewed-2x","skewed-4x","worst-case"]

# ===================== FIG 4: RQ4 routing x transport =====================
fig,axes=plt.subplots(1,2,figsize=(15,5.8))
# left: absolute latency by routing, tp1-ep4 65536, nvlink vs pcie grouped
sub=B[(B.parallel=="tp1-ep4")&(B.tokens==65536)]
x=np.arange(len(rmodes)); w=0.38
nv=[sub[(sub.routing_mode==r)&(sub.transport_condition=="nvlink_default")][LAT].values[0] for r in rmodes]
pc=[sub[(sub.routing_mode==r)&(sub.transport_condition=="no_nvls_no_p2p")][LAT].values[0] for r in rmodes]
axes[0].bar(x-w/2,nv,w,label="NVLink",color="#2a9d8f",edgecolor="white")
axes[0].bar(x+w/2,pc,w,label="PCIe fallback",color="#e76f51",edgecolor="white")
axes[0].set_xticks(x); axes[0].set_xticklabels(rmodes,rotation=25,ha="right")
axes[0].set_ylabel("Latency (ms, max rank)"); axes[0].set_title("Absolute latency by routing (tp1-ep4, 65536 tok)")
axes[0].legend(frameon=True); axes[0].grid(axis="y",alpha=.3)
# right: slowdown ratio by routing across token counts (tp1-ep4)
rpal=dict(zip(rmodes,sns.color_palette("Set2",6)))
rmark=dict(zip(rmodes,["o","s","D","^","v","P"]))
for r in rmodes:
    s=B[(B.parallel=="tp1-ep4")&(B.routing_mode==r)].sort_values("tokens")
    piv=s.pivot_table(index="tokens",columns="transport_condition",values=LAT)
    ratio=(piv["no_nvls_no_p2p"]/piv["nvlink_default"])
    axes[1].plot(ratio.index,ratio.values,marker=rmark[r],color=rpal[r],lw=2.2,ms=6,mec="white",label=r)
axes[1].axhline(1.0,color="grey",ls="--",lw=1.1)
axes[1].set_xscale("log",base=2); axes[1].set_xlabel("Tokens (log2)")
axes[1].set_ylabel("PCIe / NVLink slowdown"); axes[1].set_title("Slowdown ratio by routing (tp1-ep4)")
rqticks=[1,4,16,64,512,2048,8192,32768,65536]
axes[1].set_xticks(rqticks); axes[1].set_xticklabels([str(t) for t in rqticks],rotation=45,ha="right")
axes[1].minorticks_off(); axes[1].set_xlim(0.8,90000)
axes[1].legend(title="routing",frameon=True,fontsize=8.5,ncol=2); axes[1].grid(True,which="both",alpha=.3)
fig.suptitle("RQ4 — Routing skew slows the NVLink baseline more than the PCIe path, so the *relative* penalty shrinks (but absolute latency rises)",
             y=1.04,fontsize=12.8,fontweight="bold")
plt.tight_layout(); plt.savefig("figures/fig4_routing.png"); plt.close(); print("fig4 ok")

# ===================== FIG 5: master heatmap RQ1 (full grid to 65k, both transports) =====
A=d[d.study_name=="transport-extended-qwen3"].copy()
base=A[A.transport_condition=="nvlink_default"][["parallel","tokens",LAT]].rename(columns={LAT:"b"})
m=A.merge(base,on=["parallel","tokens"]);m["r"]=m[LAT]/m.b
fig,axes=plt.subplots(1,2,figsize=(16,5))
for ax,tc,ttl in zip(axes,["no_nvls_no_p2p","no_nvls_no_p2p_1ch"],
                     ["PCIe fallback (no NVLS/P2P)","PCIe single-channel"]):
    piv=m[m.transport_condition==tc].pivot_table(index="parallel",columns="tokens",values="r").reindex(pp_order)
    sns.heatmap(piv,annot=True,fmt=".2f",cmap="YlOrRd",vmin=1.0,vmax=5.6,linewidths=.5,
                linecolor="white",ax=ax,cbar_kws={"label":"slowdown ×"},annot_kws={"fontsize":8.5})
    ax.set_title(ttl); ax.set_xlabel("Tokens"); ax.set_ylabel("")
    ax.set_yticklabels(pp_order,rotation=0)
fig.suptitle("Interconnect-sensitivity map across the full decode→prefill range (to 65k tokens)",y=1.03,fontsize=14,fontweight="bold")
plt.tight_layout(); plt.savefig("figures/fig5_heatmap.png"); plt.close(); print("fig5 ok")
