"""Assemble the folder to upload to the Hugging Face Space: Dockerfile + README + core/dsd + server (no tests)."""
import os, shutil, zipfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
OUT = os.path.join(HERE, 'space'); shutil.rmtree(OUT, ignore_errors=True)
ign = shutil.ignore_patterns('__pycache__', '*.pyc', 'tests', '.pytest_cache', 'README.md')
shutil.copytree(os.path.join(ROOT, 'core', 'dsd'), os.path.join(OUT, 'core', 'dsd'), ignore=ign)
shutil.copytree(os.path.join(ROOT, 'server'), os.path.join(OUT, 'server'), ignore=ign)
os.remove(os.path.join(OUT, 'server', 'app_fastapi.py'))          # the Space runs the standard-library server
for f in ('Dockerfile', 'README.md'):
    shutil.copy(os.path.join(HERE, f), OUT)
z = os.path.join(HERE, 'space.zip')
with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zz:
    for r, _, fs in os.walk(OUT):
        for f in fs:
            p = os.path.join(r, f); zz.write(p, os.path.relpath(p, OUT))
print(z, sorted(zipfile.ZipFile(z).namelist()))
