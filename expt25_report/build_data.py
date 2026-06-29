import pandas as pd, glob, os

E2  = "/home/user/moe-breakdown-fresh/expt2/all_runs"
E25 = "/home/user/moe-breakdown-fresh/expt2.5/all_runs"

def load(base, canon=None):
    rows=[]
    for csv in glob.glob(f"{base}/**/results.csv", recursive=True):
        job=os.path.basename(os.path.dirname(csv)).split("_")[0]
        if canon and job not in canon: continue
        df=pd.read_csv(csv); df["job_id"]=job
        rows.append(df)
    return pd.concat(rows, ignore_index=True)

# expt2: canonical 12 jobs only
canon2=set(str(j) for j in range(5440143,5440155))
# expt2 zip needs unzip
import zipfile, tempfile
e2dir="/home/user/moe-breakdown-fresh/expt2/all_runs"
if not os.path.isdir(e2dir):
    os.makedirs(e2dir,exist_ok=True)
with zipfile.ZipFile("/home/user/moe-breakdown-fresh/expt2/all_runs.zip") as z:
    z.extractall(e2dir)

d2=load(e2dir, canon2); d2["experiment"]="expt2"
d25=load(E25); d25["experiment"]="expt2.5"

all_=pd.concat([d2,d25], ignore_index=True)
all_["parallel"]="tp"+all_.tp.astype(str)+"-ep"+all_.ep.astype(str)
print("expt2 rows:", len(d2), "| expt2.5 rows:", len(d25), "| total:", len(all_))
print("\nstudy_name counts:")
print(all_.groupby(["experiment","study_name"]).size())
print("\nexpt2.5 transport-extended transports:", sorted(d25[d25.study_name=='transport-extended-qwen3'].transport_condition.unique()))
print("expt2.5 routing modes:", sorted(d25[d25.study_name=='routing-sweep-qwen3'].routing_mode.unique()))
print("token values (expt2.5):", sorted(d25.tokens.unique()))
print("parallel:", sorted(all_.parallel.unique()))
all_.to_csv("combined.csv", index=False)
