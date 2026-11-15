40 simulated scenarios, severity 1.0, error in pixels at +10 min.

| Detections from | Ship recall | False alarms / frame | Track completeness | ID switches / ship | Forecast availability | Kalman error | Linear (5 pts) error | Linear (2 pts) error | Stationary error | 95 % ellipse coverage |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle detections (upper bound) | 1.00 | 0.00 | 1.00 | 0.04 | 1.00 | 5.5 | 6.4 | 14.7 | 16.0 | 0.96 |
| restoration + detector (ours) | 0.49 | 0.13 | 0.69 | 0.04 | 0.73 | 6.5 | 6.3 | 9.0 | 18.2 | 0.97 |
| classical filter + detector | 0.62 | 1.81 | 0.85 | 0.06 | 0.86 | 5.9 | 5.9 | 10.8 | 18.2 | 0.96 |
| detector alone | 0.02 | 0.00 | 0.16 | 0.03 | 0.23 | 17.1 | 17.2 | 17.2 | 19.3 | 1.00 |
| retrained detector (augmentation) | 0.97 | 0.29 | 1.00 | 0.01 | 1.00 | 3.8 | 3.3 | 6.4 | 16.9 | 0.96 |
