import pandas as pd, numpy as np
d=pd.read_csv("combined.csv")
LAT="latency_median_ms_max_rank"
A=d[d.study_name=="transport-extended-qwen3"]
# plateau values for abstract
for tc in ["no_nvls_no_p2p","no_nvls_no_p2p_1ch"]:
    sub=A[(A.parallel=="tp1-ep4")&(A.transport_condition.isin([tc,"nvlink_default"]))]
    piv=sub.pivot_table(index="tokens",columns="transport_condition",values=LAT)
    r=piv[tc]/piv["nvlink_default"]
    print(f"ep4 {tc}: 8192={r[8192]:.2f} 65536={r[65536]:.2f}")
# absolute worst
w=A[(A.parallel=="tp1-ep4")&(A.tokens==65536)].set_index("transport_condition")[LAT]
print(f"\nAbsolute: ep4 65536 nvlink={w['nvlink_default']:.1f}ms pcie={w['no_nvls_no_p2p']:.1f}ms 1ch={w['no_nvls_no_p2p_1ch']:.1f}ms")
