from __future__ import print_function
import sys
import mdtraj as md
from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit

if len(sys.argv) != 4:
    print('usage %s <cuda device index> <trajectory index (for output file)> <model index of starting conformation>')
    exit(1)

pdb = md.load('100-fs-peptide-400K.pdb')
forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')

system = forcefield.createSystem(pdb.topology.to_openmm(), nonbondedMethod=app.CutoffNonPeriodic,
    nonbondedCutoff=1.0*unit.nanometers, constraints=app.HBonds)
integrator = mm.LangevinIntegrator(300*unit.kelvin, 91.0/unit.picoseconds, 
    2.0*unit.femtoseconds)
integrator.setConstraintTolerance(0.00001)

platform = mm.Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed', 'CudaDeviceIndex': sys.argv[1]}
simulation = app.Simulation(pdb.topology.to_openmm(), system, integrator, platform, properties)
simulation.context.setPositions(pdb.xyz[int(sys.argv[3])])

simulation.context.setVelocitiesToTemperature(300*unit.kelvin)

nsteps = int((500*unit.nanoseconds) / (2*unit.femtoseconds))
interval = int((10*unit.picoseconds) / (2*unit.femtoseconds))

simulation.reporters.append(app.StateDataReporter(open('trajectory-%s.log' % sys.argv[2], 'w', 0),
    interval, step=True, time=True, progress=True,
    potentialEnergy=True, temperature=True, remainingTime=True,
    speed=True, totalSteps=nsteps, separator='\t'))

# equilibrate
simulation.step(int(100*unit.picoseconds / (2*unit.femtoseconds)))

# now add the trajectory reporter.
simulation.reporters.append(app.DCDReporter('trajectory-%s.dcd' % sys.argv[2], interval))
simulation.step(nsteps)
