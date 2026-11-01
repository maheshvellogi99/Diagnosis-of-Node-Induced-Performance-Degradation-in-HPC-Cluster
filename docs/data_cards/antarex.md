# Data card — Antarex HPC Fault Dataset

Telemetry classifier for Gate 5d and the a_tel likelihood (D93, D94). Never used in e-values.

## Source
- Zenodo record 2553224, DOI 10.5281/zenodo.2553224. Licence: CC-BY 4.0 (given by Mahesh, D93).
- `data/raw/antarex/Antarex.zip`, 1,632,357,471 bytes, MD5 `285e502a5e2943a5e7e5b6dd2663b06a` (equals the
  published MD5). `Readme_Antarex_Dataset.pdf`, 191,665 bytes, MD5 `02cc040ae11f648a02d5e900b80353fa` (equals
  the published MD5). SHA-256 of both in `data/raw/CHECKSUMS.txt`.
- Extracted to `data/interim/antarex/` with `bsdtar`, no new package: the two CPU-Mem archives, and from each
  only `meminfo`, `procstat`, `vmstat`, `procsensors`, `diff_perfevent`, the label file and the injection log.
  Not extracted: interrupts, `raw_counters`, `edac`, `procdiskstats`, the two HDD blocks.

## What it is (from the Readme and the files)
One node: two Intel Xeon E5-2630 v3, 128 GB RAM, CentOS 7.3, 16 cores in use, LDMS at 1 Hz (2,094 metrics
across seven plug-ins). Benchmarks and faults ran on 8 of the 16 cores. Fault tasks last about 5 minutes and
each runs in high or low intensity (49% low in the mono block). Faults do not overlap in time.

| Block | Type | Mode | Duration (Readme) | Tasks in workload file | Faults (tasks) | Benchmarks (tasks) |
|---|---|---|---|---|---|---|
| I (`CPU-MEM_mono`) | CPU-Mem | single-core | 12 days | 1,740 | leak 228, ddot 227, memeater 216, dial 212, cpufreq 210, pagefail 198 | xlinpack 123, dgemm 121, hpcc 106, stream 99 |
| III (`CPU-MEM_multi`) | CPU-Mem | multi-core | 12 days | 1,746 | cpufreq 231, pagefail 222, leak 220, dial 217, ddot 213, memeater 194 | dgemm 132, hpcc 111, xlinpack 101, stream4 56, stream8 49 |
| II (`HDD_mono`) | Hard drive | single-core | 4 days | 568 | ioerr 210, copy 208 | bonnie++ 87, iozone 55, diskcleanup 8 |
| IV (`HDD_multi`) | Hard drive | multi-core | 4 days | 551 | ioerr 206, copy 193 | iozone 76, bonnie++ 70, diskcleanup 6 |

Fault meaning per the Readme: dial = ALU interference, ddot = CPU and cache interference, memeater = memory
bandwidth interference, leak = memory leak, cpufreq = lowered frequency cap, pagefail = page-allocation failures.

## Use here (run 20261102-003936-classifier)
244 metrics after grouping (per-core metrics become sum and max over cores), 11 statistics each, 2,684
features. 60-s non-overlapping windows; windows crossing a label change are dropped (about 15%).

| fault | class (D94) | mono windows | multi windows |
|---|---|---|---|
| None | healthy | 10,140 | 9,899 |
| dial | cpu | 749 | 811 |
| ddot | cache | 831 | 786 |
| memeater | mbw | 736 | 777 |
| leak | mem | 837 | 793 |
| cpufreq | excluded from H, reported | 734 | 824 |
| pagefail | excluded from H, reported | 725 | 842 |

ioerr and copy are in the HDD blocks, which D93 says not to extract, so they have task counts above but no
windows and no classifier result.

## Limits
One machine, synthetic fault programs. Within a block the classes are separable through each program's own
activity, not through a general fault signature: see reports/phase_5.md (cross-block macro-F1 0.69 and 0.85).
