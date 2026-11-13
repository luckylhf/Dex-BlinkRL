from pathlib import Path
import tarfile,json,hashlib,time,sys
ROOT=Path(__file__).resolve().parents[1]
def snapshot(label):
 (ROOT/'backups').mkdir(exist_ok=True)
 dest=ROOT/'backups'/label
 assert not dest.with_suffix('.tar.gz').exists(), label
 paths=[p for sub in ['src','runtime','tools','configs','docs'] for p in (ROOT/sub).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
 paths += [p for p in ROOT.glob('*') if p.is_file()]
 with tarfile.open(dest.with_suffix('.tar.gz'),'w:gz') as t:
  for p in paths:t.add(p,arcname=str(p.relative_to(ROOT)))
 manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
 dest.with_suffix('.json').write_text(json.dumps({'unix':time.time(),'files':manifest,'raw_asset_manifest':'MANIFEST.json in the original release copy','restore':'Extract into a fresh release copy, then verify the backed-up source hashes. Unmodified raw assets are supplied by the release and its MANIFEST.json.'},indent=2))
 return manifest
if __name__=='__main__':print(json.dumps(snapshot(sys.argv[1]),indent=2))
