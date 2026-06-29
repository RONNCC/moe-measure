import pandas as pd, numpy as np
d=pd.read_csv("combined.csv")
LAT="latency_median_ms_max_rank"
pp_order=["tp1-ep1","tp1-ep2","tp1-ep4","tp2-ep2"]
A=d[(d.experiment=="expt2.5")&(d.study_name=="transport-extended-qwen3")]

# ====== RQ1: turnover? slowdown ratio at extended tokens ======
print("="*70);print("RQ1: TURNOVER — slowdown (no_nvls_no_p2p / nvlink) at large tokens");print("="*70)
base=A[A.transport_condition=="nvlink_default"][["parallel","tokens",LAT]].rename(columns={LAT:"b"})
m=A.merge(base,on=["parallel","tokens"]);m["r"]=m[LAT]/m.b
for tc in ["no_nvls_no_p2p","no_nvls_no_p2p_1ch"]:
    print(f"\n-- {tc} --")
    piv=m[m.transport_condition==tc].pivot_table(index="parallel",columns="tokens",values="r").reindex(pp_order)
    print(piv[[512,2048,8192,16384,32768,65536]].round(2).to_string())

# ====== RQ2: NVLS x P2P ablation @ tp1-ep4, 65536 ======
print("\n"+"="*70);print("RQ2: NVLS x P2P ablation (tp1-ep4, slowdown vs nvlink_default)");print("="*70)
for tok in [8192,65536]:
    print(f"\n-- {tok} tokens --")
    sub=A[(A.parallel=="tp1-ep4")&(A.tokens==tok)].set_index("transport_condition")[LAT]
    b=sub["nvlink_default"]
    for tc in ["nvlink_default","nvls_off","p2p_off","no_nvls_no_p2p"]:
        print(f"  {tc:20s} {sub[tc]:7.2f} ms   {sub[tc]/b:.2f}x")

# ====== RQ3: channel dose-response @ tp1-ep4, 65536 + achieved BW ======
print("\n"+"="*70);print("RQ3: CHANNEL DOSE-RESPONSE + achieved BW (tp1-ep4)");print("="*70)
chans={"no_nvls_no_p2p_1ch":1,"no_nvls_no_p2p_2ch":2,"no_nvls_no_p2p_4ch":4,"no_nvls_no_p2p_8ch":8,"no_nvls_no_p2p":"all"}
for tok in [65536]:
    print(f"\n-- {tok} tokens, tp1-ep4 --")
    sub=A[(A.parallel=="tp1-ep4")&(A.tokens==tok)].set_index("transport_condition")
    for tc,n in chans.items():
        lat=sub.loc[tc,LAT]; net=sub.loc[tc,"bucket_max_rank_network_ms"]; by=sub.loc[tc,"allgather_recv_bytes"]
        bw=by/(net*1e-3)/1e9 if net>0 else float('nan')
        print(f"  {str(n):>4} ch  lat={lat:7.2f}ms  net={net:7.2f}ms  achieved={bw:6.1f} GB/s")
    # nvlink BW reference
    nv=sub.loc["nvlink_default"]
    bw=nv["allgather_recv_bytes"]/(nv["bucket_max_rank_network_ms"]*1e-3)/1e9
    print(f"  NVLink  lat={nv[LAT]:7.2f}ms  net={nv['bucket_max_rank_network_ms']:7.2f}ms  achieved={bw:6.1f} GB/s")

# ====== RQ4: routing x transport interaction ======
print("\n"+"="*70);print("RQ4: ROUTING x TRANSPORT (tp1-ep4, 65536 tokens)");print("="*70)
B=d[(d.study_name=="routing-sweep-qwen3")&(d.parallel=="tp1-ep4")]
for tok in [8192,65536]:
    print(f"\n-- {tok} tokens: latency (ms) and PCIe/NVLink slowdown by routing --")
    sub=B[B.tokens==tok]
    piv=sub.pivot_table(index="routing_mode",columns="transport_condition",values=LAT)
    piv["slowdown"]=piv["no_nvls_no_p2p"]/piv["nvlink_default"]
    piv["alpha"]=sub.groupby("routing_mode")["alpha_observed"].first()
    print(piv.round(2).to_string())
