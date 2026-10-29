import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

fs = 50
duration = 15

t = np.arange(0, duration, 1 / fs)

frequency = 1.6
amplitude = 0.9

x = (
    amplitude * np.sin(2 * np.pi * frequency * t)
    + rng.normal(0, 0.18, len(t))
)

y = (
    0.5 * amplitude * np.sin(2 * np.pi * frequency * t + 0.7)
    + rng.normal(0, 0.18, len(t))
)

z = (
    1
    + 0.4 * amplitude * np.sin(2 * np.pi * frequency * t + 1.2)
    + rng.normal(0, 0.18, len(t))
)

df = pd.DataFrame({
    "Time": t,
    "X": x,
    "Y": y,
    "Z": z
})

df.to_csv("walking_sample_imu.csv", index=False)

print("Walking sensor CSV created successfully!")
print("Rows:", len(df))
print(df.head())