TYRETRACK MONDAY UPDATE - SOURCE PACKAGE

Four changes:
1. Production auto schedule Monday-Friday, 05:00-17:05 Brisbane local time.
2. No-count dashboard incidents are calculated from each current Brisbane production day, starting fresh at 05:00; weekend alerts are disabled. Historical counts remain saved.
3. Stop Counting and Close App prompt for four-digit operator PIN. Automatic scheduled stop is exempt. Initial PIN is configured from the requested value on first successful PIN check and saved as a salted PBKDF2 hash in SQLite state. IMPORTANT: Initial PIN is present in source code for bootstrap, so change it before making this repository public. A four-digit PIN cannot protect against Windows Task Manager, process kill, or PC shutdown.
4. Smooth (no boxes) shows the unannotated live camera stream; Show Boxes (AI matched) shows the matched analysed frame with boxes.

Source compile and static assertions passed in this environment. Windows EXE build, camera stream, live counting and operator GUI testing HAVE NOT been performed here.

Replace repository contents with these files, keeping their paths, and run the existing windows-portable GitHub Action. Preserve last known-working EXE as fallback. Verify first 20-50 tyres physically on Monday.
