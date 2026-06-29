import pandas as pd, numpy as np
d=pd.read_csv("combined.csv")
LAT="latency_median_ms_max_rank"
pp_order=["tp1-ep1","tp1-ep2","tp1-ep4","tp2-ep2"]

# ============ VALIDATION ============
print("="*70); print("VALIDATION / SANITY CHECKS"); print("="*70)
# 1. reproducibility: expt2 vs expt2.5 overlap on shared conditions (nvlink, no_nvls_no_p2p, 1ch) uniform tokens 1-8192
shared_tc=["nvlink_default","no_nvls_no_p2p","no_nvls_no_p2p_1ch"]
e2=d[(d.experiment=="expt2")]
e25a=d[(d.experiment=="expt2.5")&(d.study_name=="transport-extended-qwen3")]
merge=e2.merge(e25a, on=["transport_condition","parallel","tokens"], suffixes=("_e2","_e25"))
merge=merge[merge.transport_condition.isin(shared_tc)]
merge["pct_diff"]=(merge[LAT+"_e25"]-merge[LAT+"_e2"])/merge[LAT+"_e2"]*100
print(f"\n[Reproducibility] expt2 vs expt2.5-A on {len(merge)} shared cells:")
print(f"  median |%diff| = {merge.pct_diff.abs().median():.1f}%, p95 |%diff| = {merge.pct_diff.abs().quantile(.95):.1f}%, max = {merge.pct_diff.abs().max():.1f}%")

# 2. control invariance: tp1-ep1 should be flat across transports AND routing
ctrl=e25a[e25a.parallel=="tp1-ep1"]
base=ctrl[ctrl.transport_condition=="nvlink_default"].set_index("tokens")[LAT]
print("\n[Control invariance] tp1-ep1 transport-extended, ratio to nvlink across 8 transports:")
rr=[]
for tc in e25a.transport_condition.unique():
    s=ctrl[ctrl.transport_condition==tc].set_index("tokens")[LAT]
    rr.extend((s/base).dropna().values)
print(f"  ratio range [{min(rr):.3f}, {max(rr):.3f}], mean {np.mean(rr):.3f}")

# routing should not matter at tp1-ep1
rsw=d[(d.study_name=="routing-sweep-qwen3")&(d.parallel=="tp1-ep1")&(d.transport_condition=="nvlink_default")]
piv=rsw.pivot_table(index="tokens",columns="routing_mode",values=LAT)
spread=(piv.max(axis=1)-piv.min(axis=1))/piv.mean(axis=1)*100
print(f"\n[Routing invariance @ tp1-ep1] max spread across routing modes: {spread.max():.1f}% (should be ~0)")

# network bucket ~0 at ep1
net1=e25a[e25a.parallel=="tp1-ep1"]["bucket_max_rank_network_ms"]
print(f"[ep1 network bucket] max network_ms at tp1-ep1 = {net1.max():.3f} (should be ~0)")
