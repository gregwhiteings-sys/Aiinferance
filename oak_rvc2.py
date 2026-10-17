"""Adapters for the two supplied Luxonis 416 RVC2 exports (DepthAI v2).
Device performs inference; host decodes distance-grid heads, masks and ByteTrack.
Geometry: 480x270 full scene -> 416x234 thumbnail + 91px top/bottom pad.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
import cv2
SIZE=416
PAD=91
CONTENT_HEIGHT=234
FRAME_SIZE=(480,270)

def load_config(root,kind):
    return json.loads((Path(root)/kind/'config.json').read_text())['model']

def validate_models(dai,root):
    root=Path(root)
    hashes=json.loads((root/'sha256.json').read_text())
    for relative,digest in hashes.items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError('Compiled model/config integrity mismatch: '+relative)
    for kind in ('tyre','foreign'):
        config=load_config(root,kind)
        blob=dai.OpenVINO.Blob(str(root/kind/config['metadata']['path']))
        inputs=list(blob.networkInputs.values())
        if len(inputs)!=1 or list(inputs[0].dims)!=[416,416,3,1] or 'U8' not in str(inputs[0].dataType):
            raise ValueError('Unexpected compiled input '+kind)
        expected={o['name']:o['shape'] for o in config['outputs']}
        if set(blob.networkOutputs)!=set(expected):raise ValueError('Compiled layer names differ '+kind)
        for name,shape in expected.items():
            if list(blob.networkOutputs[name].dims)!=list(reversed(shape)):
                raise ValueError('Compiled tensor shape mismatch '+kind+'/'+name)
    return True

def build_pipeline(dai,root,recording=False,foreign_enabled=True):
    root=Path(root);validate_models(dai,root)
    p=dai.Pipeline();camera=p.create(dai.node.ColorCamera)
    camera.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camera.setPreviewSize(*FRAME_SIZE);camera.setPreviewKeepAspectRatio(True)
    camera.setInterleaved(False);camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camera.setFps(15)
    camera.setIspScale(1,3);camera.setVideoSize(640,360)
    def stream(name,out,size=4):
        node=p.create(dai.node.XLinkOut);node.setStreamName(name)
        node.input.setBlocking(False);node.input.setQueueSize(size);out.link(node.input)
    stream('live',camera.preview,1)
    if recording:
        encoder=p.create(dai.node.VideoEncoder)
        encoder.setDefaultProfilePreset(15,dai.VideoEncoderProperties.Profile.H264_MAIN)
        encoder.setBitrateKbps(1500);encoder.setKeyframeFrequency(15);encoder.setNumBFrames(0)
        encoder.input.setBlocking(False);encoder.input.setQueueSize(4)
        camera.video.link(encoder.input);stream('recording',encoder.bitstream,30)
    manip=p.create(dai.node.ImageManip)
    manip.initialConfig.setResizeThumbnail(SIZE,SIZE)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip.setMaxOutputFrameSize(SIZE*SIZE*3)
    manip.inputImage.setBlocking(False);manip.inputImage.setQueueSize(1)
    camera.preview.link(manip.inputImage)
    for kind in ('tyre','foreign') if foreign_enabled else ('tyre',):
        cfg=load_config(root,kind)
        nn=p.create(dai.node.NeuralNetwork)
        nn.setBlobPath(str(root/kind/cfg['metadata']['path']))
        nn.setNumInferenceThreads(1);nn.input.setBlocking(False);nn.input.setQueueSize(1)
        manip.out.link(nn.input)
        stream('nn' if kind=='tyre' else 'foreign_nn',nn.out,8)
        stream('rgb' if kind=='tyre' else 'foreign_rgb',nn.passthrough,4)
    return p

def read_layers(packet,kind):
    suffix='yolov8' if kind=='tyre' else 'yolov6r2'
    shapes={f'output{i}_{suffix}':(1,7,s,s) for i,s in enumerate((52,26,13),1)}
    if kind=='tyre':
        shapes.update({f'output{i}_masks':(1,32,s,s) for i,s in enumerate((52,26,13),1)})
        shapes['protos_output']=(1,32,104,104)
    out={}
    for name,shape in shapes.items():
        values=np.asarray(packet.getLayerFp16(name),dtype=np.float32)
        if values.size!=int(np.prod(shape)):raise ValueError('Malformed compiled layer '+name)
        out[name]=values.reshape(shape)
    return out

def unletterbox(image):
    if image.shape[:2]!=(416,416):raise ValueError('Unexpected OAK thumbnail dimensions')
    return cv2.resize(image[PAD:PAD+CONTENT_HEIGHT],FRAME_SIZE)

def decode(raw,confidence=.1,iou=.5,agnostic=False,with_masks=False):
    """Decode Luxonis anchor-free LTRB distances around grid+0.5.
    Score is max class probability; channel 4 is the export candidate score.
    Metadata n_prototypes=128 in this export conflicts with actual 32-channel
    tensors: mask dimensions come from actual tensor shapes, never that field.
    """
    rows=[];coeffs=[]
    suffix='yolov8' if 'output1_yolov8' in raw else 'yolov6r2'
    for i,stride in enumerate((8,16,32),1):
        s=SIZE//stride;a=np.asarray(raw[f'output{i}_{suffix}'],np.float32)
        if a.shape!=(1,7,s,s):raise ValueError('Unexpected detection head shape')
        a=a[0];scores=a[5:7].max(0);classes=a[5:7].argmax(0)
        valid=np.isfinite(a).all(0)&(a[4]>confidence)&(scores>=confidence)&(scores<=1.001)
        for y,x in zip(*np.nonzero(valid)):
            l,t,r,b=a[:4,y,x]
            if min(l,t,r,b)<0:continue
            xy=np.array([(x+.5-l)*stride,(y+.5-t)*stride,(x+.5+r)*stride,(y+.5+b)*stride],np.float32)
            xy[[0,2]]=np.clip(xy[[0,2]],0,SIZE)
            xy[[1,3]]=np.clip(xy[[1,3]],PAD,PAD+CONTENT_HEIGHT)
            if xy[2]<=xy[0] or xy[3]<=xy[1]:continue
            rows.append([*xy,float(scores[y,x]),int(classes[y,x])])
            if with_masks:coeffs.append(np.asarray(raw[f'output{i}_masks'])[0,:,y,x])
    if not rows:return np.empty((0,6),np.float32),np.empty((0,270,480),np.uint8) if with_masks else None
    rows=np.asarray(rows,np.float32);kept=[]
    for cl in ([None] if agnostic else (0,1)):
        ids=np.arange(len(rows)) if cl is None else np.flatnonzero(rows[:,5]==cl)
        if not len(ids):continue
        box=rows[ids,:4].copy();box[:,2:]-=box[:,:2]
        sel=np.asarray(cv2.dnn.NMSBoxes(box.tolist(),rows[ids,4].tolist(),confidence,iou),int).ravel()
        kept.extend(ids[sel].tolist())
    kept=sorted(kept,key=lambda i:-rows[i,4])[:300]
    chosen=rows[kept].copy();masks=None
    if with_masks:
        proto=np.asarray(raw['protos_output'],np.float32)
        if proto.shape!=(1,32,104,104):raise ValueError('Unexpected prototype dimensions')
        cs=np.asarray(coeffs,np.float32)[kept]
        if cs.shape[1]!=proto.shape[1]:raise ValueError('Mask coefficient/prototype mismatch')
        logits=cs @ proto[0].reshape(32,-1)
        masks=[]
        for row,logit in zip(chosen,logits):
            mask=cv2.resize(logit.reshape(104,104),(416,416))>0
            x1,y1,x2,y2=row[:4];yy,xx=np.ogrid[:416,:416]
            mask &= (xx>=x1)&(xx<x2)&(yy>=y1)&(yy<y2)
            masks.append(cv2.resize(mask[PAD:PAD+CONTENT_HEIGHT].astype(np.uint8),FRAME_SIZE,interpolation=cv2.INTER_NEAREST))
        masks=np.asarray(masks,np.uint8).reshape(-1,270,480)
    chosen[:,[0,2]]*=480/416
    chosen[:,[1,3]]=(chosen[:,[1,3]]-PAD)*270/234
    return chosen,masks

class OakTracker:
    names={0:'passenger',1:'4x4'}
    def __init__(self):
        from ultralytics.trackers.byte_tracker import BYTETracker
        self.tracker=BYTETracker(SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,
            new_track_thresh=.25,track_buffer=30,match_thresh=.8,fuse_score=True))
        self.predictor=SimpleNamespace(trackers=[self.tracker])
    def track_oak(self,frame,raw,confidence,iou,agnostic):
        from ultralytics.engine.results import Results,Boxes
        import torch
        boxes,masks=decode(raw,min(confidence,.1),iou,agnostic,with_masks=True)
        tracks=self.tracker.update(Boxes(boxes,frame.shape[:2]),frame)
        tracked=tracks[:,:7] if len(tracks) else np.empty((0,7),np.float32)
        selected=masks[tracks[:,7].astype(int)] if len(tracks) else np.empty((0,270,480),np.uint8)
        return Results(frame,path='OAK',names=self.names,boxes=torch.from_numpy(tracked.copy()),
                       masks=torch.from_numpy(selected.copy()))
