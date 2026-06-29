import pandas as pd, numpy as np
d=pd.read_csv("combined.csv")
LAT="latency_median_ms_max_rank"
A=d[d.study_name=="transport-extended-qwen3"]
B=d[d.study_name=="routing-sweep-qwen3"]
def L(study,pp,tok,tc,rm=None):
    s=d[(d.study_name==study)&(d.parallel==pp)&(d.tokens==tok)&(d.transport_condition==tc)]
    if rm: s=s[s.routing_mode==rm]
    return s[LAT].values[0]
ok=[]
def chk(name,val,lo,hi):
    p = lo<=val<=hi
    ok.append(p); print(f"  [{'OK' if p else 'XX'}] {name}: {val:.3f} (expect {lo}-{hi})")

print("RQ1 plateau:")
chk("ep4 PCIe 8192", L('transport-extended-qwen3','tp1-ep4',8192,'no_nvls_no_p2p')/L('transport-extended-qwen3','tp1-ep4',8192,'nvlink_default'),4.8,4.95)
chk("ep4 PCIe 65536", L('transport-extended-qwen3','tp1-ep4',65536,'no_nvls_no_p2p')/L('transport-extended-qwen3','tp1-ep4',65536,'nvlink_default'),5.1,5.25)

print("RQ2 ablation ep4 65536:")
b=L('transport-extended-qwen3','tp1-ep4',65536,'nvlink_default')
chk("nvls_off", L('transport-extended-qwen3','tp1-ep4',65536,'nvls_off')/b,0.98,1.02)
chk("p2p_off", L('transport-extended-qwen3','tp1-ep4',65536,'p2p_off')/b,1.0,1.10)
chk("both_off", L('transport-extended-qwen3','tp1-ep4',65536,'no_nvls_no_p2p')/b,5.1,5.25)
chk("ep2 p2p_off (full penalty)", L('transport-extended-qwen3','tp1-ep2',65536,'p2p_off')/L('transport-extended-qwen3','tp1-ep2',65536,'nvlink_default'),1.95,2.1)
chk("tp2ep2 p2p_off (full penalty)", L('transport-extended-qwen3','tp2-ep2',65536,'p2p_off')/L('transport-extended-qwen3','tp2-ep2',65536,'nvlink_default'),4.3,4.5)

print("RQ3 BW ratio:")
def bw(tok,tc):
    s=A[(A.parallel=="tp1-ep4")&(A.tokens==tok)&(A.transport_condition==tc)].iloc[0]
    return s["allgather_recv_bytes"]/s["bucket_max_rank_network_ms"]
chk("BW ratio 8192", bw(8192,"nvlink_default")/bw(8192,"no_nvls_no_p2p"),14,17)
chk("BW ratio 65536", bw(65536,"nvlink_default")/bw(65536,"no_nvls_no_p2p"),20,22)
# channel spread
ch=["no_nvls_no_p2p_1ch","no_nvls_no_p2p_2ch","no_nvls_no_p2p_4ch","no_nvls_no_p2p_8ch"]
lats=[L('transport-extended-qwen3','tp1-ep4',65536,c) for c in ch]
chk("channel spread %", (max(lats)-min(lats))/np.mean(lats)*100,0,12)

print("RQ4 routing 65536 ep4:")
def rr(rm): return L('routing-sweep-qwen3','tp1-ep4',65536,'no_nvls_no_p2p',rm)/L('routing-sweep-qwen3','tp1-ep4',65536,'nvlink_default',rm)
chk("uniform slowdown", rr("uniform"),5.0,5.2)
chk("worst-case slowdown", rr("worst-case"),2.7,2.9)
chk("zipfian slowdown", rr("zipfian"),3.3,3.5)
chk("worst-case abs PCIe (ms)", L('routing-sweep-qwen3','tp1-ep4',65536,'no_nvls_no_p2p','worst-case'),148,155)
chk("uniform abs PCIe (ms)", L('routing-sweep-qwen3','tp1-ep4',65536,'no_nvls_no_p2p','uniform'),125,132)
chk("worst-case NVLink (ms)", L('routing-sweep-qwen3','tp1-ep4',65536,'nvlink_default','worst-case'),52,56)

print("Absolute headline:")
chk("ep4 65536 nvlink ms", L('transport-extended-qwen3','tp1-ep4',65536,'nvlink_default'),24,26)
chk("ep4 65536 pcie ms", L('transport-extended-qwen3','tp1-ep4',65536,'no_nvls_no_p2p'),128,131)

print(f"\n=== {sum(ok)}/{len(ok)} checks passed ===")
