import os, numpy as np
try:
 import torch
 import torch.nn as nn
 import joblib
except Exception:
 torch=None

class GRUClassifier(nn.Module):
    def __init__(self):
        super().__init__(); self.gru=nn.GRU(8,48,batch_first=True,bidirectional=True); self.fc=nn.Sequential(nn.Linear(96,48),nn.ReLU(),nn.Dropout(.25),nn.Linear(48,3))
    def forward(self,x): o,_=self.gru(x); return self.fc(o[:,-1,:])

class TemporalDrowsinessModel:
    def __init__(self, model_path='models/drowsiness_gru_bootstrap.pt', scaler_path='models/feature_scaler.joblib'):
        self.ready=False; self.history=[]
        if torch is None or not os.path.exists(model_path) or not os.path.exists(scaler_path): return
        ck=torch.load(model_path,map_location='cpu',weights_only=False); self.model=GRUClassifier(); self.model.load_state_dict(ck['state_dict']); self.model.eval(); self.scaler=joblib.load(scaler_path); self.classes=ck['classes']; self.ready=True
    def predict(self, feat):
        if not self.ready: return None
        self.history.append(np.asarray(feat,dtype='float32'))
        self.history=self.history[-30:]
        if len(self.history)<30: return None
        x=np.stack(self.history); x=self.scaler.transform(x).astype('float32')
        with torch.no_grad(): prob=torch.softmax(self.model(torch.tensor(x)[None,...]),dim=1).numpy()[0]
        return {'label':self.classes[int(prob.argmax())], 'probabilities':prob.tolist()}
