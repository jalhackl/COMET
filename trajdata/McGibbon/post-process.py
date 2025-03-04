import os
import glob
import mdtraj as md
import numpy as np

top = md.load('fs-peptide.pdb')
for fn in glob.glob('trajectory-*.dcd'):
    print(fn)

    t = md.load(fn, top=top)
    # the stride is 10 picoseconds
    t.time = np.arange(t.n_frames) * 10
    t = t[::5]
    
    t.save(os.path.join('strided', os.path.splitext(fn)[0] + '.xtc'))
