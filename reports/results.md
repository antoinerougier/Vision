Clean images (no degradation): AP@0.5 = **0.999** (clean-trained detector), **0.994** (augmentation-trained detector)

| Degradation | Severity | Detector alone | + classical filter | + restoration (ours) | Retrained detector (augmentation) | PSNR before → after |
|---|---|---|---|---|---|---|
| sar_speckle | 0.5 | 0.517 | 0.899 | 0.930 | 0.986 | 16.9 → 28.0 dB |
| sar_speckle | 1.0 | 0.083 | 0.614 | 0.825 | 0.944 | 13.2 → 25.3 dB |
| sar_speckle | 1.3 (unseen) | 0.035 | 0.398 | 0.678 | 0.859 | 12.2 → 24.3 dB |
| low_light | 0.5 | 0.740 | 0.975 | 0.968 | 1.000 | 18.6 → 25.8 dB |
| low_light | 1.0 | 0.002 | 0.811 | 0.713 | 0.986 | 13.5 → 20.9 dB |
| low_light | 1.3 (unseen) | 0.000 | 0.516 | 0.063 | 0.870 | 12.9 → 18.4 dB |
| fog | 0.5 | 0.114 | 0.961 | 0.972 | 0.994 | 8.4 → 24.0 dB |
| fog | 1.0 | 0.000 | 0.682 | 0.585 | 0.993 | 6.0 → 22.2 dB |
| fog | 1.3 (unseen) | 0.000 | 0.459 | 0.174 | 0.994 | 5.4 → 21.5 dB |
| thermal | 0.5 | 0.545 | 0.679 | 0.744 | 0.995 | 18.5 → 28.2 dB |
| thermal | 1.0 | 0.120 | 0.232 | 0.067 | 0.962 | 15.8 → 25.9 dB |
| thermal | 1.3 (unseen) | 0.029 | 0.067 | 0.009 | 0.840 | 14.8 → 24.3 dB |
