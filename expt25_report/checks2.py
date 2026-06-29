import pandas as pd, numpy as np
d=pd.read_csv("combined.csv")
LAT="latency_median_ms_max_rank"
A=d[(d.study_name=="transport-extended-qwen3")]

# achieved BW: note network_ms is summed over PROFILED iters, not 1. Check scale.
# We computed bytes/(network_ms) treating net as per-iter. Verify how many profiled iters.
# Compare achieved-BW RATIO (nvlink vs pcie) which is robust to per-iter normalization.
print("RQ3 robustness: BW *ratio* nvlink/pcie (normalization-independent):")
for tok in [8192,65536]:
    sub=A[(A.parallel=="tp1-ep4")&(A.tokens==tok)].set_index("transport_condition")
    def bw(tc): 
        return sub.loc[tc,"allgather_recv_bytes"]/sub.loc[tc,"bucket_max_rank_network_ms"]
    print(f"  tok={tok}: nvlink/1ch BW ratio = {bw('nvlink_default')/bw('no_nvls_no_p2p_1ch'):.1f}x, nvlink/allch = {bw('nvlink_default')/bw('no_nvls_no_p2p'):.1f}x")

# channel scaling: does latency change at all 1->8ch? 
print("\nRQ3 channel latency (tp1-ep4, 65536):")
ch={"no_nvls_no_p2p_1ch":1,"no_nvls_no_p2p_2ch":2,"no_nvls_no_p2p_4ch":4,"no_nvls_no_p2p_8ch":8}
sub=A[(A.parallel=="tp1-ep4")&(A.tokens==65536)].set_index("transport_condition")
lats=[sub.loc[tc,LAT] for tc in ch]
print(f"  1->8ch latency range: {min(lats):.1f}-{max(lats):.1f}ms, spread {(max(lats)-min(lats))/np.mean(lats)*100:.1f}%")

# Confirm ablation holds across ALL parallel pts at 65536
print("\nRQ2 ablation across parallel points (65536, slowdown vs nvlink):")
for pp in ["tp1-ep2","tp1-ep4","tp2-ep2"]:
    sub=A[(A.parallel==pp)&(A.tokens==65536)].set_index("transport_condition")[LAT]
    b=sub["nvlink_default"]
    print(f"  {pp}: nvls_off={sub['nvls_off']/b:.2f}x  p2p_off={sub['p2p_off']/b:.2f}x  both={sub['no_nvls_no_p2p']/b:.2f}x")

# RQ1: absolute latency at 65536 - is it dominated by network now?
print("\nRQ1 bucket fraction (network+sync) tp1-ep4 65536:")
for tc in ["nvlink_default","no_nvls_no_p2p"]:
    r=A[(A.parallel=="tp1-ep4")&(A.tokens==65536)&(A.transport_condition==tc)].iloc[0]
    bcols=[c for c in d.columns if c.startswith("bucket_max_rank_") and c.endswith("_ms")]
    tot=sum(r[c] for c in bcols)
    comp=r["bucket_max_rank_gpu_compute_ms"]
    ns=(r["bucket_max_rank_network_ms"]+r["bucket_max_rank_gpu_idle_sync_ms"])/tot*100
    print(f"  {tc}: net+sync={ns:.0f}%, gpu_compute_ms={comp:.2f}")
