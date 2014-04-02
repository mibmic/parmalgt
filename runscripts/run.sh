#!/bin/sh
#SBATCH -N 1            # 1 node
#SBATCH -t 02:30:00     # run time
#SBATCH -p compute      # partition name
#SBATCH -J swf0.05      # sensible name for the job

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

L 6             # Lattice extend, we set T = L in the simulation
s 0             # s parameter for staggered
alpha 0.05      # gauge fixing parameter
taug 0.05       # Langevin integration step size
tauf 0.05       # Wilson flow integration step size
NRUN 20000     # number of configs to generate
NFLOW 30        # number of flow steps
MEAS_FREQ 100   # frequency of measurements outside flow
FLOW_MEAS_FREQ 1    # after how many flow steps do we want to measure?
FLOW_FREQ 10    # after how many Langevin steps do we want to flow?

seed $RANDOM$RANDOM    # random number seed

# reading / writing gauge configurations
read gauge       # file name for input, 'none' for cold start
write gauge      # file name for output, 'none' to omit output
EOF

/home/users/dirk/run/sigma/wflow/LocalQuenchFlow
