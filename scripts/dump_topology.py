#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


COMMANDS = {
    "nvidia_smi_L": ["nvidia-smi", "-L"],
    "nvidia_smi_topo": ["nvidia-smi", "topo", "-m"],
    "nvidia_smi_nvlink": ["nvidia-smi", "nvlink", "--status"],
    "nvidia_smi_query": [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version,pci.bus_id",
        "--format=csv,noheader",
    ],
    "lspci": ["bash", "-lc", "lspci | egrep -i 'nvidia|mellanox|infiniband' || true"],
    "ibv_devinfo": ["ibv_devinfo"],
}


def main() -> int:
    out = {}
    for name, cmd in COMMANDS.items():
        try:
            out[name] = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        except Exception as exc:
            out[name] = f"UNAVAILABLE: {exc}"
    path = Path("topology_snapshot.json")
    path.write_text(json.dumps(out, indent=2))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
