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
P05,P95="latency_p05_ms_max_rank","latency_p95_ms_max_rank"
pp_order=["tp1-ep1","tp1-ep2","tp1-ep4","tp2-ep2"]
A=d[d.study_name=="transport-extended-qwen3"].copy()

# ===================== FIG 1: RQ1 turnover — slowdown vs tokens to 65k with p5/p95 bands =====================
base=A[A.transport_condition=="nvlink_default"][["parallel","tokens",LAT]].rename(columns={LAT:"b"})
m=A.merge(base,on=["parallel","tokens"])
m["r"]=m[LAT]/m.b
m["r_lo"]=m[P05]/m.b
m["r_hi"]=m[P95]/m.b

pp_pal={"tp1-ep1":"#264653","tp1-ep2":"#2a9d8f","tp1-ep4":"#e76f51","tp2-ep2":"#8338ec"}
pp_mark={"tp1-ep1":"o","tp1-ep2":"s","tp1-ep4":"D","tp2-ep2":"^"}
pp_lab={"tp1-ep1":"tp1-ep1 (control)","tp1-ep2":"tp1-ep2","tp1-ep4":"tp1-ep4 (max EP)","tp2-ep2":"tp2-ep2"}
fig,axes=plt.subplots(1,2,figsize=(15,6),sharey=True)
for ax,tc,ttl in zip(axes,["no_nvls_no_p2p","no_nvls_no_p2p_1ch"],
                     ["PCIe fallback (no NVLS/P2P)","PCIe single-channel"]):
    sub=m[m.transport_condition==tc]
    for pp in pp_order:
        s=sub[sub.parallel==pp].sort_values("tokens")
        ax.plot(s.tokens,s.r,marker=pp_mark[pp],color=pp_pal[pp],label=pp_lab[pp],lw=2.4,ms=7,mec="white",mew=.8)
        # p5-p95 shaded band
        ax.fill_between(s.tokens,s.r_lo,s.r_hi,color=pp_pal[pp],alpha=.12)
    ax.axhline(1.0,color="grey",ls="--",lw=1.2)
    ax.axvspan(8192,65536,color="#ffd6a5",alpha=.25,lw=0)
    ax.set_xscale("log",base=2); ax.set_title(ttl); ax.set_xlabel("Tokens (log₂ scale)")
    xticks=[1,4,16,64,512,2048,8192,32768,65536]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks],rotation=45,ha="right")
    ax.minorticks_off()
    ax.set_xlim(0.8, 90000)
    ax.grid(True,which="both",alpha=.3)
axes[0].set_ylabel("Slowdown ratio  (latency / NVLink baseline)")
axes[1].text(9500,1.45,"expt2.5\nextended\nregion",fontsize=9,color="#bc6c25",ha="left",va="bottom",fontweight="bold")
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,loc="upper center",ncol=4,frameon=True,bbox_to_anchor=(0.5,1.05))
fig.suptitle("RQ1 — The slowdown plateaus; it never turns over (communication-bound across the entire prefill range)\n"
             "Shaded bands = p5–p95 per-iteration latency range; data: expt2.5-A, uniform routing, all 4 layouts",
             y=1.12,fontsize=13,fontweight="bold")
plt.tight_layout(); plt.savefig("figures/fig1_turnover.png"); plt.close(); print("fig1 ok")

# ===================== FIG 2: RQ2 NVLS x P2P ablation (grouped bars) with p5/p95 error bars =====================
abl=["nvlink_default","nvls_off","p2p_off","no_nvls_no_p2p"]
abl_lab=["NVLink\n(baseline)","NVLS off\n(P2P on)","P2P off\n(NVLS on)","both off\n(PCIe)"]
abl_col=["#2a9d8f","#90be6d","#f9c74f","#e76f51"]
fig,axes=plt.subplots(1,2,figsize=(15,5.8),sharey=False)
for ax,tok in zip(axes,[8192,65536]):
    x=np.arange(len(pp_order[1:])); w=0.2
    sub=A[A.tokens==tok]
    for i,(tc,lab,c) in enumerate(zip(abl,abl_lab,abl_col)):
        vals=[]; lo=[]; hi=[]
        for pp in pp_order[1:]:
            row=sub[(sub.parallel==pp)&(sub.transport_condition==tc)]
            vals.append(row[LAT].values[0])
            lo.append(row[P05].values[0])
            hi.append(row[P95].values[0])
        vals,lo,hi=np.array(vals),np.array(lo),np.array(hi)
        xpos=x+(i-1.5)*w
        bars=ax.bar(xpos,vals,w,label=lab,color=c,edgecolor="white",lw=.6)
        ax.errorbar(xpos,vals,yerr=[vals-lo,hi-vals],fmt="none",color="black",capsize=3,lw=1.2,capthick=1)
    ax.set_xticks(x); ax.set_xticklabels(pp_order[1:])
    ax.set_title(f"{tok:,} tokens"); ax.set_ylabel("Latency (ms, max rank, median ± p5/p95)")
    ax.grid(axis="y",alpha=.3); ax.set_yscale("log")
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,loc="upper center",ncol=4,frameon=True,bbox_to_anchor=(0.5,1.07))
fig.suptitle("RQ2 — Losing P2P-IPC causes the damage; NVLS (NVLink-SHARP) is ~free for this collective\n"
             "Data: expt2.5-A, uniform routing; error bars = p5–p95 of per-iteration latency (100 iters)",
             y=1.15,fontsize=13,fontweight="bold")
fig.text(0.5,-0.02,"Note: at tp1-ep4, P2P-off alone stays ~baseline (NVLS still carries the load); at tp1-ep2 / tp2-ep2, P2P-off alone already incurs the full penalty.",
         ha="center",fontsize=8.5,style="italic",color="#555")
plt.tight_layout(); plt.savefig("figures/fig2_ablation.png"); plt.close(); print("fig2 ok")

# ===================== FIG 3: RQ3 channel dose-response + achieved BW (tp1-ep4 only) =====================
fig,axes=plt.subplots(1,2,figsize=(15,5.8))
# left: latency vs channels at several token counts (tp1-ep4 ONLY)
chmap={"no_nvls_no_p2p_1ch":1,"no_nvls_no_p2p_2ch":2,"no_nvls_no_p2p_4ch":4,"no_nvls_no_p2p_8ch":8}
toks=[2048,8192,32768,65536]; cpal=sns.color_palette("flare",len(toks))
for tok,c in zip(toks,cpal):
    sub=A[(A.parallel=="tp1-ep4")&(A.tokens==tok)]
    xs=[1,2,4,8]; ys=[sub[sub.transport_condition==tc][LAT].values[0] for tc in chmap]
    axes[0].plot(xs,ys,marker="o",color=c,lw=2.2,ms=8,mec="white",label=f"{tok:,} tok")
    # nvlink reference dashed
    nv=sub[sub.transport_condition=="nvlink_default"][LAT].values[0]
    axes[0].axhline(nv,color=c,ls=":",lw=1.2,alpha=.7)
axes[0].set_xscale("log",base=2); axes[0].set_xticks([1,2,4,8]); axes[0].set_xticklabels([1,2,4,8])
axes[0].set_yscale("log"); axes[0].set_xlabel("NCCL_MAX_NCHANNELS (PCIe path)"); axes[0].set_ylabel("Latency (ms, max rank, median)")
axes[0].set_title("tp1-ep4: Adding PCIe channels doesn't help\n(dotted = NVLink floor, same color)")
axes[0].legend(title="tokens",frameon=True,fontsize=8.5); axes[0].grid(True,which="both",alpha=.3)
# right: achieved BW (GB/s) vs tokens, nvlink vs pcie — tp1-ep4 only
def bw(df): return df["allgather_recv_bytes"]/(df["bucket_max_rank_network_ms"]*1e-3)/1e9
sub=A[(A.parallel=="tp1-ep4")&(A.transport_condition.isin(["nvlink_default","no_nvls_no_p2p"]))].copy()
sub["bw"]=bw(sub)
tcpal={"nvlink_default":"#2a9d8f","no_nvls_no_p2p":"#e76f51"}
tclab={"nvlink_default":"NVLink","no_nvls_no_p2p":"PCIe fallback"}
for tc in ["nvlink_default","no_nvls_no_p2p"]:
    s=sub[sub.transport_condition==tc].sort_values("tokens")
    s=s[s.bucket_max_rank_network_ms>0]
    axes[1].plot(s.tokens,s.bw,marker="o",color=tcpal[tc],lw=2.4,ms=7,mec="white",label=tclab[tc])
axes[1].set_xscale("log",base=2); axes[1].set_yscale("log")
axes[1].set_xlabel("Tokens (log₂ scale)"); axes[1].set_ylabel("Achieved all-gather BW (GB/s)\n[profiler-derived; ratio robust, absolute values are not]")
axes[1].set_title("tp1-ep4: ~15–20× bandwidth cliff NVLink→PCIe")
bwticks=[1,4,16,64,512,2048,8192,32768,65536]
axes[1].set_xticks(bwticks); axes[1].set_xticklabels([str(t) for t in bwticks],rotation=45,ha="right")
axes[1].minorticks_off()
axes[1].legend(frameon=True); axes[1].grid(True,which="both",alpha=.3)
fig.suptitle("RQ3 — The bottleneck is the PCIe staging path, not channel parallelism (tp1-ep4 only)\n"
             "Data: expt2.5-A, uniform routing; BW = allgather_recv_bytes / profiler_network_time",
             y=1.04,fontsize=13,fontweight="bold")
plt.tight_layout(); plt.savefig("figures/fig3_channels_bw.png"); plt.close(); print("fig3 ok")
