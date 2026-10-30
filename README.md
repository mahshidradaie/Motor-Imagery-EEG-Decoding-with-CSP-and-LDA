# Motor-Imagery BCI: Band-pass + CSP + LDA (simulated EEG)

Left/right-hand imagery decoding from simulated 16-channel EEG (250 Hz, 3 s trials). Imagery suppresses the 10 Hz
mu rhythm over the contralateral sensorimotor source (event-related desynchronisation); sources are mixed into
sensors with ongoing alpha activity, background sources, trial-to-trial gain variability and sensor noise.
Pipeline: 8–30 Hz band-pass → Common Spatial Patterns (generalised eigenproblem, 3 filter pairs, fitted **inside** each
training fold) → log-variance features → LDA, 10-fold CV, 5 simulation seeds. Baseline: per-channel log band power + LDA.

```
pip install numpy scipy scikit-learn matplotlib
python bci_csp.py     # writes results.json and accuracy.png
```

## Results (`results.json`)
| Sensor noise | CSP + LDA | Channel log-power + LDA |
|---|---|---|
| 3 | 99.1 % | 86.5 % |
| 6 | 95.0 % | 79.2 % |
| 10 | 85.6 % | 69.6 % |

## Caveats
**Simulated data**: absolute accuracies are optimistic compared with real EEG (e.g. BCI Competition recordings) and
only show that the pipeline recovers a known effect and that spatial filtering beats raw channel power under
spatial mixing. Not validated on real recordings.
