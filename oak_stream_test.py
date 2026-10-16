"""Exercise actual capture threads with independent camera and AI packet sequences."""
import os,time,threading
os.environ['QT_QPA_PLATFORM']='offscreen'
import numpy as np
import TyreTrack_Production_Monitor as m
logs=[];m.log=logs.append;m.oak_frame_received=lambda:None;m.oak_stream_stopped=lambda:None
m.RECORD_OAK_VIDEO=False
class Packet:
 def __init__(self,seq,nn=False):self.seq=seq;self.nn=nn
 def getSequenceNum(self):return self.seq
 def getTimestamp(self):return m.dai.Clock.now()
 def getCvFrame(self):
  shape=(270,480,3) if self.seq==23 else (384,384,3)
  return np.full(shape,self.seq,dtype=np.uint8)
 def getLayerFp16(self,name):return np.full(18144,.123,dtype=np.float32)
class Q:
 def __init__(self,items):self.items=list(items)
 def tryGet(self):return self.items.pop(0) if self.items else None
class Device:
 def __init__(self,*args):
  self.queues={'live':Q([Packet(23)]),'rgb':Q([Packet(9),Packet(10)]),
               'nn':Q([Packet(10,True),Packet(99,True)])}
 def getOutputQueue(self,name,**kw):return self.queues[name]
 def close(self):pass
original=m.dai.Device;m.dai.Device=Device
source=None
try:
 source=m.OakSource('')
 ok,frame,seq=source.read();assert ok and frame.shape==(270,480,3)
 assert np.all(frame==10), 'Counting must use matched AI passthrough, never newer preview'
 assert np.allclose(source.last_read_neural_output,.123)
 deadline=time.monotonic()+1
 while source.preview_snapshot()[0] is None and time.monotonic()<deadline:time.sleep(.005)
 live,live_seq,_=source.preview_snapshot();assert live_seq==23 and np.all(live==23)
 time.sleep(.03)
 assert source.sequence==seq, 'Unmatched NN packet must not publish'
 before=len(logs);source._stream_diagnostics(source.capture_diagnostic_since+10.01,time.monotonic())
 assert len(logs)==before+1 and 'OAK stream summary:' in logs[-1]
 assert 'max_preview_loop_pause_ms=' in logs[-1]
 assert not any('recording WARNING' in x for x in logs)
finally:
 if source:source.release()
 m.dai.Device=original
assert not source.reader.is_alive() and not source.preview_reader.is_alive()
print('PASS: independent preview, exact AI sequence match, no unmatched publication, periodic timing summary, clean thread shutdown')
