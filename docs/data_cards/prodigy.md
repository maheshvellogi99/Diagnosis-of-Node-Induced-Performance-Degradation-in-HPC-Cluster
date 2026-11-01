# Data card — Prodigy artifact (Eclipse)

Secondary binary check only, reported only (D94). Not used in any e-value or in the a_tel likelihood.

## Source
- Zenodo record 8079388, DOI 10.5281/zenodo.8079388. Licence: CC-BY 4.0 (given by Mahesh, D93).
- `data/raw/prodigy/eclipse_small_prod_dataset.tar`, 44,553,216 bytes, MD5 `7e7514fe2ccd8d01e0c7b3f28b30f57b`
  (equals the published MD5). SHA-256 in `data/raw/CHECKSUMS.txt`.
- `data/raw/prodigy/fe_eclipse_tsfresh_raw_CHI_2000.json`, 211,926 bytes, MD5
  `3d8968004eee12e5d43354c860fad311` (equals the published MD5). Keys: `raw_column_names`,
  `tsfresh_column_names`.

## Contents, from the label files inside the tar
| file | rows (job-node) | jobs | nodes | classes (rows) | apps (rows) |
|---|---|---|---|---|---|
| `prod_train_label.csv` | 64 | 20 | 58 | none 64 | exa 19, lammps 17, sw4 15, sw4lite 13 |
| `prod_test_label.csv` | 176 | 53 | 141 | cpuoccupy 80, memleak 80, none 16 | sw4lite 47, sw4 45, lammps 43, exa 41 |

Columns: job_id, component_id, app_name, anom_name, anom_input, binary_anom.
`prod_train_data.hdf` and `prod_test_data.hdf` are HDF5.

**Correction to D94's wording:** the test labels hold cpuoccupy as well as memleak, so the artifact is
not "healthy vs memleak only". The training file is healthy only.

## Status
`UNKNOWN — needs check`: feature counts, sampling and window length inside the HDF files. Reading them
needs `h5py` or `tables`, which are not installed; CLAUDE.md rule 4 requires asking first. The secondary
binary check is therefore **not run**.
