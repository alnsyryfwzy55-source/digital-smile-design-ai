#!/usr/bin/env bash
# Run after `flutter create` (CI does both): bundles the web app and sets the Android permissions and name.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf assets/web && mkdir -p assets && cp -r ../docs assets/web
rm -f assets/web/sw.js                       # no service worker inside the app: the files are already on the phone
test -s assets/web/teeth_unet_v2.onnx || { echo "teeth_unet_v2.onnx missing"; exit 1; }
python3 - <<'PY'
import re, pathlib
p = pathlib.Path('android/app/src/main/AndroidManifest.xml'); s = p.read_text()
perms = ['android.permission.INTERNET', 'android.permission.CAMERA']
for perm in perms:
    if perm not in s:
        s = s.replace('<application', f'<uses-permission android:name="{perm}"/>\n    <application', 1)
if 'android.hardware.camera' not in s:
    s = s.replace('<application', '<uses-feature android:name="android.hardware.camera" android:required="false"/>\n    <application', 1)
s = re.sub(r'android:label="[^"]*"', 'android:label="تصميم الابتسامة"', s, count=1)
if 'usesCleartextTraffic' not in s:                 # the app talks to its own server on http://localhost
    s = s.replace('<application', '<application android:usesCleartextTraffic="true"', 1)
p.write_text(s)
for g in pathlib.Path('android/app').glob('build.gradle*'):
    t = g.read_text()
    t = re.sub(r'minSdk\s*=\s*flutter\.minSdkVersion', 'minSdk = 24', t)
    t = re.sub(r'minSdkVersion\s+flutter\.minSdkVersion', 'minSdkVersion 24', t)
    g.write_text(t)
print(s[:900])
PY
