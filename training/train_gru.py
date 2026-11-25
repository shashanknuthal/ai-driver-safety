"""Train the temporal drowsiness model on REAL labeled feature sequences.
CSV columns: sequence_id,label,frame,ear,mar,abs_yaw,pitch_down,eye_closure_duration,yawn_duration,blink_rate,yawn_rate
Labels: alert, low_vigilance, drowsy
"""
import argparse, json, joblib, numpy as np, pandas as pd, torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

FEATURES=['ear','mar','abs_yaw','pitch_down','eye_closure_duration','yawn_duration','blink_rate','yawn_rate']
CLASSMAP={'alert':0,'low_vigilance':1,'drowsy':2}
class GRUClassifier(nn.Module):
    def __init__(self):
        super().__init__(); self.gru=nn.GRU(8,48,batch_first=True,bidirectional=True); self.fc=nn.Sequential(nn.Linear(96,48),nn.ReLU(),nn.Dropout(.25),nn.Linear(48,3))
    def forward(self,x): o,_=self.gru(x); return self.fc(o[:,-1,:])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); ap.add_argument('--epochs',type=int,default=50); ap.add_argument('--out',default='models/drowsiness_gru.pt'); a=ap.parse_args()
    df=pd.read_csv(a.csv).sort_values(['sequence_id','frame'])
    groups=[]
    for sid,g in df.groupby('sequence_id'):
        lab=str(g.label.iloc[0]).lower();
        if lab not in CLASSMAP: continue
        z=g[FEATURES].dropna().values.astype('float32')
        if len(z)>=30: groups.append((sid,lab,z[-30:]))
    rng=np.random.default_rng(42); rng.shuffle(groups); n=len(groups); tr=int(.7*n); va=int(.15*n)
    train, val, test=groups[:tr],groups[tr:tr+va],groups[tr+va:]
    scaler=StandardScaler().fit(np.concatenate([z for _,_,z in train],axis=0))
    def pack(gs): return np.stack([scaler.transform(z) for _,_,z in gs]).astype('float32'), np.array([CLASSMAP[l] for _,l,_ in gs])
    Xtr,ytr=pack(train); Xv,yv=pack(val); Xte,yte=pack(test)
    model=GRUClassifier(); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); lossf=nn.CrossEntropyLoss(); best=1e9; state=None; patience=0
    for e in range(a.epochs):
        model.train(); opt.zero_grad(); loss=lossf(model(torch.tensor(Xtr)),torch.tensor(ytr)); loss.backward(); opt.step()
        model.eval();
        with torch.no_grad(): vl=lossf(model(torch.tensor(Xv)),torch.tensor(yv)).item()
        if vl<best: best=vl; state={k:v.detach().clone() for k,v in model.state_dict().items()}; patience=0
        else: patience+=1
        if patience>=8: break
    model.load_state_dict(state); model.eval();
    with torch.no_grad(): p=model(torch.tensor(Xte)).argmax(1).numpy()
    print(classification_report(yte,p,target_names=list(CLASSMAP.keys()),digits=4))
    print(confusion_matrix(yte,p))
    torch.save({'state_dict':model.state_dict(),'feature_names':FEATURES,'classes':list(CLASSMAP.keys()),'sequence_length':30},a.out)
    joblib.dump(scaler,a.out.replace('.pt','_scaler.joblib'))
if __name__=='__main__': main()
