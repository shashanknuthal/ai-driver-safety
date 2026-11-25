"""Collect labeled webcam feature sequences for real training.
Run one session per label and subject. Press q to stop.
Examples:
  python collect_data.py --label alert --subject s01
  python collect_data.py --label low_vigilance --subject s01
  python collect_data.py --label drowsy --subject s01
"""
import argparse, csv, time, cv2, yaml
from features.face_processor import FaceProcessor

FEATURES=['sequence_id','subject_id','label','frame','timestamp','ear','mar','abs_yaw','pitch_down','eye_closure_duration','yawn_duration','blink_rate','yawn_rate']
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--label',required=True,choices=['alert','low_vigilance','drowsy']); ap.add_argument('--subject',required=True); ap.add_argument('--out',default='datasets/real_features.csv'); a=ap.parse_args()
    cfg=yaml.safe_load(open('config.yaml')); cap=cv2.VideoCapture(cfg['camera']['index']); p=FaceProcessor(cfg)
    import os; os.makedirs('datasets',exist_ok=True); exists=os.path.exists(a.out)
    seq=f'{a.subject}_{a.label}_{int(time.time())}'; start=time.time(); frames=0
    f=open(a.out,'a',newline='',encoding='utf8'); w=csv.DictWriter(f,fieldnames=FEATURES)
    if not exists: w.writeheader()
    print('Record now. Follow the label condition naturally. q=stop.')
    while True:
        ok,frame=cap.read()
        if not ok: break
        frame=cv2.flip(frame,1); r=p.process(frame); now=time.time()
        if r.get('face'):
            row={'sequence_id':seq,'subject_id':a.subject,'label':a.label,'frame':frames,'timestamp':now,'ear':r['ear'],'mar':r['mar'],'abs_yaw':abs(r['yaw']),'pitch_down':r['pitch']}
            # Feature durations are not available directly here; keep zero placeholders for real retraining baseline.
            row.update({'eye_closure_duration':0.0,'yawn_duration':0.0,'blink_rate':0.0,'yawn_rate':0.0}); w.writerow(row); f.flush(); frames+=1
        cv2.putText(frame,f'LABEL: {a.label} | frames: {frames}',(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,0),2); cv2.imshow('Data Collection',frame)
        if cv2.waitKey(1)&0xff==ord('q'): break
    f.close(); cap.release(); cv2.destroyAllWindows(); p.close(); print('Saved',frames,'frames to',a.out)
if __name__=='__main__': main()
