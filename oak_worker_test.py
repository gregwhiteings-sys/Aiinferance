"""Run the real OAK counting worker with synthetic device packets, without camera or DB writes."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import numpy as np,time
from PyQt5.QtCore import QCoreApplication
import TyreTrack_Production_Monitor as m
qt=QCoreApplication.instance() or QCoreApplication([])
logged=[];statuses=[]
m.log_count=lambda *args:logged.append(args)
m.oak_session_allowed=lambda manual:True
m.YOLO=lambda *a,**k:(_ for _ in ()).throw(AssertionError('OAK must not load CPU tyre YOLO'))
ys=list(range(30,13,-1))+list(range(13,31))
class SimulatedOak:
 def __init__(self,ip,foreign_enabled=False):self.index=0;self.last_read_frame_at=time.monotonic()
 def read(self):
  if self.index==len(ys):worker.running=False;return False,None,None
  raw={f'output{i}_yolov8':np.zeros((1,7,s,s),np.float32) for i,s in enumerate((52,26,13),1)}
  raw.update({f'output{i}_masks':np.zeros((1,32,s,s),np.float32) for i,s in enumerate((52,26,13),1)})
  raw['protos_output']=np.zeros((1,32,104,104),np.float32)
  raw['output1_yolov8'][0,:,ys[self.index],26]=[5,4,5,4,.95,.95,.01]
  self.last_read_neural_output=raw;self.index+=1;self.last_read_frame_at=time.monotonic()
  return True,np.zeros((270,480,3),dtype=np.uint8),self.index
 def release(self):pass
m.OakSource=SimulatedOak
worker=m.CounterWorker('oak','','Bottom to Top',.35,.5)
worker.status.connect(statuses.append)
worker.run()
assert not any(s.startswith('ERROR') for s in statuses),statuses
assert worker.session_count==1 and len(logged)==1,(worker.session_count,logged,statuses)
assert worker.session_type_counts['passenger']==1
print('PASS: actual OAK worker bypasses CPU YOLO, tracks packets, counts single crossing once and reconciles Passenger total')
