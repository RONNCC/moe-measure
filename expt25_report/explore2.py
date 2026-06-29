import pandas as pd, numpy as np
d=pd.read_csv("combined.csv")
LAT="latency_median_ms_max_rank"

# Repro outlier: where is the 40% diff?
e2=d[d.experiment=="expt2"]
e25a=d[(d.experiment=="expt2.5")&(d.study_name=="transport-extended-qwen3")]
m=e2.merge(e25a,on=["transport_condition","parallel","tokens"],suffixes=("_e2","_e25"))
m=m[m.transport_condition.isin(["nvlink_default","no_nvls_no_p2p","no_nvls_no_p2p_1ch"])]
m["pct"]=(m[LAT+"_e25"]-m[LAT+"_e2"])/m[LAT+"_e2"]*100
print("Top repro diffs:")
print(m.reindex(m.pct.abs().sort_values(ascending=False).index)[["transport_condition","parallel","tokens",LAT+"_e2",LAT+"_e25","pct"]].head(8).to_string(index=False))
print(f"\nAll big diffs at small tokens? token distribution of |pct|>10%:")
print(m[m.pct.abs()>10].tokens.value_counts().to_string())

# Routing invariance at ep1: which routing modes, which tokens?
rsw=d[(d.study_name=="routing-sweep-qwen3")&(d.parallel=="tp1-ep1")&(d.transport_condition=="nvlink_default")]
piv=rsw.pivot_table(index="tokens",columns="routing_mode",values=LAT)
print("\ntp1-ep1 latency by routing mode (nvlink):")
print(piv.round(3).to_string())
print("\nalpha_observed at tp1-ep1 by routing:")
print(rsw.pivot_table(index="tokens",columns="routing_mode",values="alpha_observed").round(2).head(4).to_string())
