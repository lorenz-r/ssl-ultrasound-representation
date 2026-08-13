"""WI-0 probe: stream real NV-Raw2Insights-US frames and measure everything the
plan depends on — IQ shape/dtype, acquisition scalars, grid extents, the
reciprocity residual, onset structure, and the bmode vs bmode_variant_2 relation.

Each frame is a ~265 MB download, so this is run once and its JSON output checked
in at tests/data/wi0_results.json for tests/test_nv_frame.py to assert against.

    N_FRAMES=4 python scripts/wi0_probe.py
"""
import os, time, json

import numpy as np
from datasets import load_dataset

N_FRAMES = int(os.environ.get("N_FRAMES","4"))
ds = load_dataset("nvidia/NV-Raw2Insights-US", split="validation", streaming=True).with_format("numpy")
out=[]; t0w=time.time()
for idx, s in enumerate(ds):
    if idx >= N_FRAMES: break
    re = np.asarray(s["iq_real"], dtype=np.float32)
    im = np.asarray(s["iq_imag"], dtype=np.float32)
    print(f"[{idx}] loaded {re.shape} {re.dtype} in {time.time()-t0w:.0f}s", flush=True)
    S = (re + 1j*im).astype(np.complex64); del re, im
    r={"idx":idx, "iq_shape":list(S.shape), "iq_dtype":"float32"}

    ST = np.transpose(S,(1,0,2))
    D = S-ST; nS=float(np.linalg.norm(S))
    r["recip_residual"]=float(np.linalg.norm(D)/nS)
    off=~np.eye(S.shape[0],dtype=bool)
    r["recip_residual_offdiag"]=float(np.linalg.norm(D[off])/np.linalg.norm(S[off]))
    # magnitude-only symmetry (phase-insensitive check)
    aS=np.abs(S); aST=np.abs(ST)
    r["recip_residual_mag"]=float(np.linalg.norm(aS-aST)/np.linalg.norm(aS))
    del D, ST, aST

    thr=0.01*aS.max(); ab=aS>thr
    onset=np.where(ab.any(-1), ab.argmax(-1), aS.shape[-1]-1).astype(np.float32)
    n=S.shape[0]
    r["onset_diag_mean"]=float(np.mean(np.diag(onset)))
    r["onset_offdiag_mean"]=float(onset[off].mean())
    r["onset_asym_mean_abs"]=float(np.abs(onset-onset.T).mean())
    sep=np.abs(np.subtract.outer(np.arange(n),np.arange(n)))
    r["onset_vs_sep"]={int(d): float(onset[sep==d].mean()) for d in (0,10,40,90,150,179)}
    e0=(aS**2).sum((1,2)); e1=(aS**2).sum((0,2))
    r["axis0_energy_cv"]=float(e0.std()/e0.mean()); r["axis1_energy_cv"]=float(e1.std()/e1.mean())
    r["axis_energy_corr"]=float(np.corrcoef(e0,e1)[0,1])
    r["axis_energy_reldiff"]=float(np.abs(e0-e1).mean()/e0.mean())
    prof=(aS**2).sum((0,1)); prof/=prof.max()
    r["last_sample_above_1e-4"]=int(np.max(np.where(prof>1e-4)[0]))
    r["iq_rms"]=float(np.sqrt((aS**2).mean()))
    del aS, S

    ep=np.asarray(s["elpos"],dtype=np.float32)
    r["elpos_shape"]=list(ep.shape); r["pitch_um"]=float(np.diff(ep[0]).mean()*1e6)
    for k in ("fs","fc","fd","t0","c0","phase_error","frame_idx"): r[k]=float(s[k])
    bm=np.asarray(s["bmode"],dtype=np.float32); bmf=np.asarray(s["bmode_focused"],dtype=np.float32)
    seg=np.asarray(s["segmentation_map"]); sos=np.asarray(s["sound_speed_map"],dtype=np.float32)
    r["bmode_shape"]=list(bm.shape); r["seg_shape"]=list(seg.shape)
    r["seg_unique"]=sorted(np.unique(seg).tolist()); r["sos_shape"]=list(sos.shape)
    r["bmode_extent"]=[float(v) for v in np.asarray(s["bmode_extent"])]
    a=20*np.log10(np.maximum(bm,1e-9)); a-=a.max()
    b=20*np.log10(np.maximum(bmf,1e-9)); b-=b.max()
    m=(a>-60)|(b>-60)
    r["bmode_vs_variant2_corr"]=float(np.corrcoef(a[m],b[m])[0,1])
    r["bmode_vs_variant2_mad_db"]=float(np.abs(a[m]-b[m]).mean())
    r["sharp_bmode"]=float((np.diff(bm,axis=1)**2).mean()/(bm**2).mean())
    r["sharp_variant2"]=float((np.diff(bmf,axis=1)**2).mean()/(bmf**2).mean())
    r["sos_range"]=[float(sos.min()),float(sos.max())]
    out.append(r); print("  ->", json.dumps({k:r[k] for k in ("recip_residual","recip_residual_mag","onset_diag_mean","onset_offdiag_mean","onset_asym_mean_abs")}), flush=True)
json.dump(out, open("/tmp/wi0/wi0_results.json","w"), indent=1)
print("WI0_DONE", len(out))
