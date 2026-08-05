from pathlib import Path
def safe_path(root,requested):
    root=Path(root).resolve(); path=(root/requested).resolve()
    if path!=root and root not in path.parents: raise ValueError('path outside review root')
    return path
def allowed_origin(host): return host in {'127.0.0.1','localhost'}
