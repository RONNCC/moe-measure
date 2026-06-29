import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
import seaborn as sns
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":150,"savefig.bbox":"tight",
  "axes.titleweight":"bold","axes.titlesize":13,"axes.labelsize":11.5,
  "legend.fontsize":9,"xtick.labelsize":9.5,"ytick.labelsize":9.5,"font.family":"DejaVu Sans"})
d=pd.read_csv("combined.csv")
A=d[d.study_name=="transport-extended-qwen3"]

# Buckets that actually carry signal (drop the all-zero ones for clarity, keep an "other")
bspec=[("bucket_max_rank_network_ms","Network (NCCL collective)","#e76f51"),
       ("bucket_max_rank_gpu_idle_sync_ms","GPU idle — waiting on sync","#f4a261"),
       ("bucket_max_rank_cpu_native_ms","CPU native (launch / runtime)","#577590"),
       ("bucket_max_rank_mem_transfer_ms","Host<->Device mem transfer","#9b5de5"),
       ("bucket_max_rank_allocator_ms","Allocator","#43aa8b")]

# ===== FIG 6: stacked breakdown vs tokens, tp1-ep4, NVLink vs PCIe side by side =====
fig,axes=plt.subplots(1,2,figsize=(15,6.2),sharey=False)
toks=[1,8,64,512,2048,8192,32768,65536]
for ax,(tc,ttl) in zip(axes,[("nvlink_default","NVLink (baseline)"),("no_nvls_no_p2p","PCIe fallback (no NVLS/P2P)")]):
    sub=A[(A.parallel=="tp1-ep4")&(A.transport_condition==tc)].set_index("tokens")
    sub=sub.loc[toks]
    # normalize to % of profiled time for composition view
    mat=np.vstack([sub[c].values for c,_,_ in bspec])
    totals=mat.sum(axis=0); pct=mat/totals*100
    bottom=np.zeros(len(toks)); x=np.arange(len(toks))
    for i,(c,lab,col) in enumerate(bspec):
        ax.bar(x,pct[i],bottom=bottom,color=col,label=lab,edgecolor="white",lw=.5)
        bottom+=pct[i]
    ax.set_xticks(x); ax.set_xticklabels([str(t) for t in toks],rotation=45,ha="right")
    ax.set_title(ttl); ax.set_ylim(0,100); ax.set_xlabel("Tokens"); ax.set_ylabel("% of profiled GPU/CPU events")
    ax.grid(axis="y",alpha=.3)
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,loc="upper center",ncol=3,frameon=True,bbox_to_anchor=(0.5,1.10))
fig.suptitle("Profiled time composition vs. token count (tp1-ep4): the Network bucket is what grows under degradation",y=1.16,fontsize=13.5,fontweight="bold")
fig.text(0.5,-0.04,"Composition of torch.profiler events over 5 profiled iters, max rank. Buckets overlap in wall-clock time (CPU launch/sync run concurrently with GPU work), so this shows RELATIVE shifts, not an additive latency budget. "
                   "GPU-compute/memory buckets register ~0: the fused-MoE expert kernels are not separately attributed here. The robust, wall-clock signal is in Fig. 1–3.",
         ha="center",fontsize=7.6,style="italic",color="#555",wrap=True)
plt.tight_layout(); plt.savefig("figures/fig6_breakdown.png"); plt.close(); print("fig6 ok")

# ===== FIG 7: ABSOLUTE network-bucket time vs tokens, all parallel pts, nvlink vs pcie =====
fig,axes=plt.subplots(1,2,figsize=(15,5.6),sharey=True)
pp_pal={"tp1-ep2":"#2a9d8f","tp1-ep4":"#e76f51","tp2-ep2":"#8338ec"}
pp_mark={"tp1-ep2":"s","tp1-ep4":"D","tp2-ep2":"^"}
for ax,(tc,ttl) in zip(axes,[("nvlink_default","NVLink (baseline)"),("no_nvls_no_p2p","PCIe fallback")]):
    for pp in ["tp1-ep2","tp1-ep4","tp2-ep2"]:
        s=A[(A.parallel==pp)&(A.transport_condition==tc)].sort_values("tokens")
        ax.plot(s.tokens,s.bucket_max_rank_network_ms,marker=pp_mark[pp],color=pp_pal[pp],lw=2.3,ms=7,mec="white",label=pp)
    ax.set_xscale("log",base=2); ax.set_yscale("log"); ax.set_title(ttl); ax.set_xlabel("Tokens (log2)")
    xt=[1,4,16,64,512,2048,8192,32768,65536]; ax.set_xticks(xt); ax.set_xticklabels([str(t) for t in xt],rotation=45,ha="right")
    ax.minorticks_off(); ax.grid(True,which="both",alpha=.3)
axes[0].set_ylabel("Network bucket time (ms, max rank,\n5 profiled iters)")
axes[0].legend(title="parallel",frameon=True)
fig.suptitle("Absolute time in the Network (NCCL) bucket: ~20× higher on PCIe, scaling with token count",y=1.02,fontsize=13.5,fontweight="bold")
plt.tight_layout(); plt.savefig("figures/fig7_network_abs.png"); plt.close(); print("fig7 ok")
