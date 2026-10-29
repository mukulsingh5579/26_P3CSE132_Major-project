import numpy as np, pandas as pd, joblib
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ACTIVITIES=["Walking","Sitting","Standing","Turning","Lying","Stairs"]

# For the demo, synthetic IMU-like windows are generated.
# Replace this function with real labeled IMU CSV loading for the final project.
def make_demo_data(n_per_class=600, seed=42):
    rng=np.random.default_rng(seed); rows=[]; fs=50
    for label in ACTIVITIES:
        for _ in range(n_per_class):
            t=np.arange(2.56,step=1/fs)
            if label=="Walking":
                f=rng.uniform(1.2,2.0); a=rng.uniform(.6,1.2)
                ax=a*np.sin(2*np.pi*f*t)+rng.normal(0,.18,len(t))
                ay=.5*a*np.sin(2*np.pi*f*t+.7)+rng.normal(0,.18,len(t))
                az=1+.4*a*np.sin(2*np.pi*f*t+1.2)+rng.normal(0,.18,len(t))
            elif label=="Stairs":
                f=rng.uniform(1,1.7)
                ax=1.1*np.sin(2*np.pi*f*t)+rng.normal(0,.25,len(t))
                ay=.7*np.sin(2*np.pi*f*t+.4)+rng.normal(0,.22,len(t))
                az=1.2+.7*np.sin(2*np.pi*f*t+1)+rng.normal(0,.25,len(t))
            elif label=="Turning":
                f=rng.uniform(.35,.8)
                ax=.8*np.sin(2*np.pi*f*t)+rng.normal(0,.2,len(t))
                ay=.8*np.cos(2*np.pi*f*t)+rng.normal(0,.2,len(t))
                az=1+rng.normal(0,.18,len(t))
            elif label=="Sitting":
                ax=rng.normal(0,.08,len(t)); ay=rng.normal(0,.08,len(t)); az=1+rng.normal(0,.08,len(t))
            elif label=="Standing":
                ax=rng.normal(0,.11,len(t)); ay=rng.normal(0,.11,len(t)); az=1+rng.normal(0,.11,len(t))
            else:
                ax=rng.normal(0,.07,len(t)); ay=1+rng.normal(0,.07,len(t)); az=rng.normal(0,.07,len(t))
            rows.append((ax,ay,az,label))
    return rows

def features(ax,ay,az,fs=50):
    mag=np.sqrt(ax**2+ay**2+az**2); c=mag-mag.mean()
    fft=np.abs(np.fft.rfft(c)); freq=np.fft.rfftfreq(len(c),1/fs)
    dom=freq[np.argmax(fft[1:])+1]
    peaks,_=find_peaks(c,prominence=max(c.std()*.5,1e-6))
    return [mag.mean(),mag.std(),mag.min(),mag.max(),np.percentile(mag,25),
            np.percentile(mag,75),dom,len(peaks)/(len(mag)/fs),
            ax.std(),ay.std(),az.std()]

rows=make_demo_data()
X=np.array([features(a,b,c) for a,b,c,_ in rows])
y=np.array([lab for *_,lab in rows])
le=LabelEncoder(); y=le.fit_transform(y)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced")
model.fit(Xtr,ytr); pred=model.predict(Xte)
print("Accuracy:",accuracy_score(yte,pred))
print(classification_report(yte,pred,target_names=le.classes_,digits=3))
print("Confusion matrix:\n",confusion_matrix(yte,pred))
joblib.dump({"model":model,"label_encoder":le},"human_activity_model.pkl")
