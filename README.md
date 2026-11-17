# Modeling Gait With a Passive Exoskeleton

This project reproduces the figures and modeled joint responses in Keyan Dong's mechanical system dynamics course report. The data cover 191 strides from ten walking trials before, during and after use of a passive lower-limb exoskeleton.

## Reproduce the figures

With Python 3.12 and the four packages in `requirements.txt`, run the following commands in the repository root.

```sh
python -m pip install -r requirements.txt
python analyze.py
```

The analysis runs offline once its dependencies are installed. It writes six figures and two CSV files to `results/`. The CSVs contain the two-second model comparison under identical baseline torques and the fifth-cycle joint trajectories for each walking condition. Use `--output DIRECTORY` to choose another location.

`data/phase_profiles.csv` contains the stride-phase means and standard deviations behind the plantar-load and muscle-deformation figures. The three rows of `data/condition_summary.csv` contain the stride counts, timing, load and torque statistics underlying the report. Individual recordings and per-stride data are excluded.

`dynamics.py` converts bilateral mean muscle deformation into torque indices and integrates the joint equations. The thigh and shank carry the added frame mass. At the ankle, the model uses a 7.5 N/mm heel spring and assumes a 0.17 m moment arm. The resulting angles are model predictions. `figures.py` draws the experimental and modeled comparisons.

## Analyze recordings in the same format

`preprocess.py` builds the same two condition-level tables from muscle and plantar-force recordings. A recording directory contains two subdirectories.

| Subdirectory | Expected input |
| --- | --- |
| `肌肉形变/` | One UTF-8 CSV per trial, with an `Index` sample column at 100 Hz and six columns named `1` through `6` for the paired GM, RF and MG channels. The reader skips instrument metadata above the column header. |
| `足底力/` | Two whitespace-delimited files per trial ending in `-L.txt` and `-R.txt`. Each sample row holds 54 sensor values (18 three-axis sensors) followed by elapsed time in milliseconds. |

Name each trial consistently across its three files, for example `pre1.csv`, `pre1-L.txt` and `pre1-R.txt`. The analysis expects baseline (`pre`), device (`exo`) and post-removal (`post`) trials with numbered identifiers. It normalizes muscle channels across all trials, detects strides between consecutive right-foot load peaks, and resamples each accepted stride to 101 phase points. Strides receive equal weight in the deformation and torque profiles. Stride-period and load summaries give each trial equal weight.

```sh
python preprocess.py recordings --output summaries
python analyze.py --data summaries --output results
```
