HUMAN ACTIVITY MONITORING — 60% PROTOTYPE

Project: Human Activity Monitoring for Early Neurodegenerative Disease Assessment
Student: Mukul Singh (2410030823)

Implemented:
1. IMU-style 3-axis acceleration input
2. Windowing / feature extraction
3. Mean, standard deviation, percentiles, dominant frequency and peak-rate features
4. Random Forest activity classifier
5. Accuracy, precision, recall, F1 and confusion matrix
6. Saved trained model
7. Demo prediction with confidence

Demo classes: Walking, Sitting, Standing, Turning, Lying, Stairs
Train/test split: 80/20
Demo windows: 3600
Features/window: 11

DEMO RESULT
Accuracy: 99.31%
Macro Precision: 99.31%
Macro Recall: 99.31%
Macro F1: 99.31%

IMPORTANT:
This execution uses synthetic IMU-like data because no real sensor dataset was included in the
uploaded Phase-I report. It is a software prototype result, NOT a clinical/medical validation.
For the next phase, replace the demo generator with a real labeled wearable dataset.
