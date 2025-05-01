#!/ban/bash
#
#SBATCH --job-name=wildfire_train
#SBATCH --output=wildfire_train.log
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=10000
#SBATCH --mail-user=neil.mewada@sjsu.edu
#SBATCH --mail-type=END
python3 train.py

