from __future__ import annotations
import time, numpy as np
from agentipc.state.hub import StateHub

def run_e4(*, repeat):
    rows=[]
    for transport in ("inproc","shm"):
        for i in range(repeat):
            hub=StateHub(transport=transport); arr=np.arange(64,dtype=np.float32); t=time.perf_counter(); ref=hub.put_array(arr,kind="formal_e4",summary="shared memory comparison"); resolved=hub.resolve_array(ref); latency=(time.perf_counter()-t)*1000; rows.append({"experiment":transport,"repeat":i,"latency_ms":latency,"state_bytes":int(ref.nbytes),"transfer_metrics":{"transport":transport,"resolved":bool(np.array_equal(arr,resolved))}}); hub.release(ref); hub.close()
    return rows
