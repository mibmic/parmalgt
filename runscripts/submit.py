import os
from subprocess import Popen

# tau values to use
TAUVALS = (.0015, .003, .006, .008, .012, .016, .024)
NVALS = (64, 32, 16, 12, 8, 6, 4)
NMAX = 6
TIME = TAUVALS[0]*NVALS[0]
EXE = "/home/users/dirk/run/wflow/LocalQuenchFlow"

for t, n in zip(TAUVALS, NVALS):
    assert(t*n == TIME)

SHF = """#!/bin/sh
#SBATCH -N 1           # 1 node
#SBATCH -t 00:%i:00   # run time
#SBATCH -p %s         # partition name
#SBATCH -J wf_%f      # sensible name for the job

cat <<EOF > flow_ipt
# Sample configuration file
#
# Please note:
#   o Parameters are CASE SENSITIVE
#   o Parameters must be given in the form 
#        parameter value
#     with one ore more spaces in between
#   o The order of the parameters does not matter
#   o A hash sign is interpreted as a comment

L 4             # Lattice extend, we set T = L in the simulation
s 0             # s parameter for staggered
alpha 0.05      # gauge fixing parameter
taug %f       # Langevin integration step size
tauf %f       # Wilson flow integration step size
NRUN 320000     # number of configs to generate
NFLOW %i        # number of flow steps
FLOW_MEAS_FREQ %i    # frequency of measurements during flow
FLOW_FREQ %i     # after how many Langevin steps do we want to flow?
MEAS_FREQ 320000 # fequ. of measurements outside flow
seed $RANDOM$RANDOM    # random number seed

# reading / writing gauge configurations
read gauge       # file name for input, 'none' for cold start
write gauge      # file name for output, 'none' to omit output
EOF

../LocalQuenchFlow
"""

if __name__ == "__main__":
    for tau, n in zip(TAUVALS, NVALS):
        path = "tau%s" % str(tau)[1:]
        try:
            os.mkdir(path)
        except OSError:
            pass # don't mind if directory exists
        of = open(path + "/run.sh", "w")
        of.write(SHF % (60, "compute", tau, tau, tau , n*NMAX, n, 10))
        Popen(("sbatch", "run.sh"), cwd=path)
