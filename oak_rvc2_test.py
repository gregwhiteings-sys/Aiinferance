"""Real compiled blob validation, pipeline construction, raw-head and mask tests."""
from pathlib import Path
import numpy as np,depthai as dai
from oak_rvc2 import validate_models,build_pipeline,decode,unletterbox,OakTracker,read_layers
ROOT=Path(__file__).parent/'rvc2_models'
assert validate_models(dai,ROOT)
for foreign in (False,True):
 for recording in (False,True):
  p=build_pipeline(dai,ROOT,recording,foreign)
  cameras=[n for n in p.getAllNodes() if isinstance(n,dai.node.ColorCamera)]
  assert cameras[0].getIspSize()==cameras[0].getVideoSize()==(640,360)
  assert cameras[0].getPreviewSize()==(480,270)
  networks=[n for n in p.getAllNodes() if isinstance(n,dai.node.NeuralNetwork)]
  assert len(networks)==(2 if foreign else 1)
def empty(tyre=True):
 suffix='yolov8' if tyre else 'yolov6r2'
 raw={f'output{i}_{suffix}':np.zeros((1,7,s,s),np.float32) for i,s in enumerate((52,26,13),1)}
 if tyre:
  raw.update({f'output{i}_masks':np.zeros((1,32,s,s),np.float32) for i,s in enumerate((52,26,13),1)})
  raw['protos_output']=np.zeros((1,32,104,104),np.float32)
 return raw
def put(raw,x=26,y=26,c=0):
 suffix='yolov8' if 'output1_yolov8' in raw else 'yolov6r2'
 raw['output1_'+suffix][0,:,y,x]=[5,5,5,5,.95,.95 if c==0 else .01,.95 if c==1 else .01]
raw=empty();put(raw)
raw['output1_masks'][0,0,26,26]=1;raw['protos_output'][0,0]=1
boxes,masks=decode(raw,with_masks=True)
# grid center 26.5 * 8 =212, edges 172,252. Map 416px letterbox to 480x270.
np.testing.assert_allclose(boxes[0,:4],[172*480/416,(172-91)*270/234,252*480/416,(252-91)*270/234],rtol=1e-5)
assert masks.shape==(1,270,480) and masks.sum()>0
put(raw,x=27);boxes,_=decode(raw);assert len(boxes)==1,'NMS must suppress overlapping same class'
put(raw,x=27,c=1);assert len(decode(raw)[0])==2,'Class aware NMS'
assert len(decode(raw,agnostic=True)[0])==1
assert decode(empty())[0].shape==(0,6)
bad=empty();bad['output1_yolov8']=np.zeros(3)
try:decode(bad);raise AssertionError('Malformed shape accepted')
except ValueError:pass
pad=empty();put(pad,x=20,y=1);assert len(decode(pad)[0])==0
nan=empty();put(nan);nan['output1_yolov8'][0,0,26,26]=np.nan;assert len(decode(nan)[0])==0
rgb=np.zeros((416,416,3),np.uint8);rgb[91:325,:,2]=255
frame=unletterbox(rgb);assert frame.shape==(270,480,3) and np.all(frame[:,:,2]==255)
tracker=OakTracker();ids=[]
for y in range(30,17,-1):
 r=empty();put(r,y=y);r['output1_masks'][0,0,y,26]=1;r['protos_output'][0,0]=1
 result=tracker.track_oak(frame,r,.35,.5,False)
 assert result.boxes.id is not None and result.masks is not None
 ids.append(int(result.boxes.id[0]))
assert len(set(ids))==1
# Confirm both actual class IDs route through alarm matching, independently.
from foreign_alarm import CLASSES,Confirmation
assert CLASSES==('wheel_rim','person')
for cl in (0,1):
 confirm=Confirmation();event=None
 for t in (0,.2,.4):
  f=empty(False);put(f,c=cl);rows,_=decode(f,.45)
  event=confirm.update([dict(class_id=int(r[5]),confidence=float(r[4]),box=r[:4].tolist()) for r in rows],t)
 assert event['class_id']==cl
print('PASS: actual compiled hash/input/output metadata; 4 pipeline combinations; full FOV; distance-grid decode; NMS/class IDs; masks; malformed/NaN/padding rejection; ByteTrack continuity; both alarm classes')
