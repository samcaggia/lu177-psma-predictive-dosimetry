# output/

Empty in the repository. GATE water kernels are written here:

| Folder | Produced by |
|---|---|
| `vsv/` | `scripts/gate/lu177_water_validation.py vsv` (4 mm Lu-177 voxel S-value kernel in water; read by nb04 and `src/psma_mc.py`) |
| `sphere/` | `scripts/gate/lu177_water_validation.py sphere` |
| `vsv_native/` | `scripts/gate/lu177_native_kernel.py` on the three PET grids (via `scripts/monte_carlo/run_mc_retest.py`; read by `src/psma_mc.py`) |

The GATE scripts score photon steps with the DoseActor `hit_type = split` by default; the older `random` scoring
(`--hit-type random`) inflates the photon tail of water kernels (`docs/results/vsv_kernel_lu177_water.png`).
