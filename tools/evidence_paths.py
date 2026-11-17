"""Resolve historical evidence paths inside the extracted package, never the old checkout."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TAGS=("Dex-BlinkWalk","Dex-BlinkRL")
def recorded_path(value,root=None):
 root=Path(root).resolve() if root is not None else ROOT.resolve()
 p=Path(value)
 if not p.is_absolute():candidate=root/p
 elif p.is_relative_to(root):candidate=p
 elif any(t in p.parts for t in TAGS):
  tag=next(t for t in TAGS if t in p.parts)
  index=max(i for i,x in enumerate(p.parts) if x==tag)
  candidate=root.joinpath(*p.parts[index+1:])
 else:raise ValueError("Recorded path is outside the packaged Dex-BlinkWalk evidence: "+str(p))
 resolved=candidate.resolve()
 if not resolved.is_relative_to(root):raise ValueError("Evidence path escapes package: "+str(p))
 if not resolved.exists():raise FileNotFoundError(resolved)
 return resolved
