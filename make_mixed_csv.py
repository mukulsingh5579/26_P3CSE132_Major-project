import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

fs = 50
window_seconds = 2.56
samples_per_window = int(fs * window_seconds)

activities = [
    "Walking",
    "Sitting",
    "Standing",
    "Turning",
    "Lying",
    "Stairs"
]

all_data = []

for activity in activities:

    for window in range(5):

        t = np.arange(samples_per_window) / fs

        if activity == "Walking":

            f = 1.6
            a = 0.9

            x = (
                a * np.sin(2 * np.pi * f * t)
                + rng.normal(0, 0.18, len(t))
            )

            y = (
                0.5 * a * np.sin(
                    2 * np.pi * f * t + 0.7
                )
                + rng.normal(0, 0.18, len(t))
            )

            z = (
                1 + 0.4 * a * np.sin(
                    2 * np.pi * f * t + 1.2
                )
                + rng.normal(0, 0.18, len(t))
            )

        elif activity == "Stairs":

            f = 1.3

            x = (
                1.1 * np.sin(2 * np.pi * f * t)
                + rng.normal(0, 0.25, len(t))
            )

            y = (
                0.7 * np.sin(
                    2 * np.pi * f * t + 0.4
                )
                + rng.normal(0, 0.22, len(t))
            )

            z = (
                1.2 + 0.7 * np.sin(
                    2 * np.pi * f * t + 1
                )
                + rng.normal(0, 0.25, len(t))
            )

        elif activity == "Turning":

            f = 0.55

            x = (
                0.8 * np.sin(2 * np.pi * f * t)
                + rng.normal(0, 0.2, len(t))
            )

            y = (
                0.8 * np.cos(2 * np.pi * f * t)
                + rng.normal(0, 0.2, len(t))
            )

            z = (
                1 + rng.normal(0, 0.18, len(t))
            )

        elif activity == "Sitting":

            x = rng.normal(0, 0.08, len(t))
            y = rng.normal(0, 0.08, len(t))
            z = 1 + rng.normal(0, 0.08, len(t))

        elif activity == "Standing":

            x = rng.normal(0, 0.11, len(t))
            y = rng.normal(0, 0.11, len(t))
            z = 1 + rng.normal(0, 0.11, len(t))

        elif activity == "Lying":

            x = rng.normal(0, 0.07, len(t))
            y = 1 + rng.normal(0, 0.07, len(t))
            z = rng.normal(0, 0.07, len(t))

        for i in range(len(t)):

            all_data.append([
                activity,
                window + 1,
                t[i],
                x[i],
                y[i],
                z[i]
            ])


df = pd.DataFrame(
    all_data,
    columns=[
        "Activity",
        "Window",
        "Time",
        "X",
        "Y",
        "Z"
    ]
)

df.to_csv(
    "mixed_imu_demo.csv",
    index=False
)

print("================================")
print("Mixed IMU CSV created!")
print("================================")
print("Activities:", activities)
print("Windows per activity:", 5)
print("Total rows:", len(df))
print("File: mixed_imu_demo.csv")