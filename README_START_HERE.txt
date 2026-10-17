TYRETRACK — DUAL RVC2 INFERENCE SOURCE UPDATE

This is SOURCE for a Windows build, not a new prebuilt EXE.
Both new compiled models are included under rvc2_models with SHA-256 checks.
The working PC tyre model is retained; OAK now uses the new segmentation blob.
The old six-class foreign demo is replaced at runtime by wheel_rim/person.
The small foreign training set is synthetic and does not cover general debris.

BUILD
Extract this ZIP and upload its CONTENTS to your existing GitHub repository,
including the hidden .github/workflows folder. Use the included windows-portable.yml
workflow. Run it manually, then download its Windows portable artifact.
Existing build steps validate app logic and smoke-test the generated executable.
No Windows executable has been built or tested on this Linux workspace.

PHYSICAL TEST (before replacing a working yard installation)
1. Keep a backup of the current working portable app and its data/settings.
2. Close other apps using the OAK-D S2. Connect the camera to the test PC/network.
3. Run RUN_OAK_HARDWARE_TEST.bat from the new portable folder.
   It runs both networks, independent preview, full-view 640x360 H264 recording
   and host output/mask decoding for 60 seconds.
   It writes oak_hardware_report.json and oak_probe_recording MP4(s) beside the app.
   This camera test runs independently of production hours and does not count tyres,
   email reports, clear alarms or write to the production database.
4. Send the JSON report back for review. A passed stream test confirms outputs,
   not detection accuracy, alarm sound, count accuracy, or safe operating speed.
5. If the dual test stalls, run the tyre-only probe:
   TyreTrack.exe --oak-hardware-test --tyre-only --seconds 60 --output oak_tyre_only.json
   For a PoE IP, add --ip 192.168.x.x with the actual camera address.
6. Use the app's OAK manual test mode for the end-to-end count/GUI/alarm test.
   Manual tests may write counts/incidents; use an isolated app copy and test data.
   Enable the foreign detector BEFORE Start. Tune the orange belt ROI.
   Configure an admin password. Verify alarm sound/banner persists until admin reset.
   Never place a person on or near a moving conveyor to test the detector.

RUNTIME DETAILS
Both compiled networks use full-scene 416x416 BGR planar thumbnails; normalization
and channel reversal are embedded by the compiler. Live preview is 480x270; hardware
H264 recording is 640x360. Each branch pairs outputs with its own passthrough sequence.
CPU performs tensor decoding, segmentation mask reconstruction, ByteTrack, counting,
reporting and alarm state; OAK performs both neural inferences and H264 encoding.
Actual mask tensor dimension is 32, despite converter metadata saying 128 prototypes.
The adapter validates and uses the compiled tensor shapes.
The frame interval slider applies only to PC video tests; live OAK uses each newly
received foreign result once. Three spatially matching detections confirm the alarm.
An ROI-excluded object will not alarm. A foreign stream silent >6 seconds shows a fault.
Existing confidence settings are retained; new installs default to 45% foreign confidence.
Neither network automatically detects all unknown objects.
Wheel/rim alarms DO NOT automatically subtract tyres from counts. Existing counting
rules are retained. A later exclusion policy requires an explicit operating decision.
The old legacy blob/adapter remain bundled only for regression checks, not the new
live pipeline. Older notes referring to six-class demo or 384 input are superseded here.

LOCAL VERIFICATION
See integration_checks.json, source-smoke.json, foreign_cpu_validation.json.
Checks cover compiled metadata/pipeline construction, distance-grid/NMS/masks,
sequence matching, tracking/counting, recording, reporting, GUI, alarm class mapping
and authenticated reset. No physical camera inference or Windows packaging ran here.

AUTOMATIC PRODUCTION SCHEDULE UPDATE
Every app launch enables Auto connect and count and clears a previous manual-stop
pause. At 05:00 Brisbane (UTC+10), or immediately when launched during shift,
the app selects OAK and starts production automatically. It retries failed camera
connections every 60 seconds. Production stops at 17:05. Outside this window it
waits for the next 05:00. This applies every day. The app must remain open and the
PC awake; this update does not install Windows login startup or wake the PC.
The Stop button still pauses automatic restart for the rest of the day while
that app session remains open. Re-enabling Auto or reopening the app resumes it.
Manual after-hours OAK testing remains available and separate from production.
