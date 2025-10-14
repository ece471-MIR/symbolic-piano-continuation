from pathlib import Path
import miditok

data_root_path = Path.cwd() / "data" / "aria" / "aria-midi-v1-pruned-ext"

files = list((data_root_path / "data" / "aa").iterdir())

config = miditok.TokenizerConfig(
    pitch_range=(0, 127), use_velocities=False, encode_ids_splits="no"
)

remi = miditok.REMI(config)

"""
ts = []
for idx, x in enumerate(tqdm.tqdm((data_root_path / "data").iterdir())):
    for f in x.iterdir():
        m = symusic.Score(f)
        if not m.time_signatures:
            ts.append("4/4")
        else:
            ts.append(m.time_signatures)
"""

tokens = []

for f in files:

    pass

import pdb

pdb.set_trace()
