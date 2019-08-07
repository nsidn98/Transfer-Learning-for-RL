# Note: all paths referenced here are relative to the Docker container.
#
# Add the Nvidia drivers to the path
export PATH="/usr/local/nvidia/bin:$PATH"
export HOME="/storage/home/sidnayak"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
# Tools config for CUDA, Anaconda installed in the common /tools directory
source /tools/config.sh
# Activate your environment
source activate py35
# Change to the directory in which your code is present
cd /storage/home/sidnayak/Transfer-Matching-Networks/src/algos/cifar_experiment/


# Run the code. The -u option is used here to use unbuffered writes
# so that output is piped to the file as and when it is produced.

# python -u Agent_detect.py --load_model=1 --epoch=10 &> out
# python -u ssd_random.py --epoch=20 &> outputs/out_random_newReward
python -u cifarTest.py --tensorboard=1 --rm_runs=1 &> out.txt

