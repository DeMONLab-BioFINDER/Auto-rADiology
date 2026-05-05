#!/bin/bash
# SLURM batch job script for Berzelius (raw multi-output regression)

#SBATCH -A berzelius-2026-31
#SBATCH --gpus=1
#SBATCH -t 00:45:00
#SBATCH -J tau-raw-multi
#SBATCH -o ../logs/slurm-%j.out
#SBATCH -e ../logs/slurm-%j.err

# Load environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet

# Stage training data to node-local scratch for faster I/O.
SRC_DATA_DIR="/proj/berzelius-2024-156/users/x_nadpi/data/data/tau_raw"
SRC_DEMO_CSV="/proj/berzelius-2024-156/users/x_nadpi/data/demo.csv"
SCRATCH_INPUT="/scratch/local/ai_pet_input"

if [[ ! -d /scratch/local ]]; then
  echo "[stage] ERROR: /scratch/local is not available on this node."
  echo "[stage] Run this script only on allocated compute nodes."
  exit 1
fi

echo "[stage] hostname: $(hostname)"
echo "[stage] preparing scratch under: ${SCRATCH_INPUT}"
mkdir -p "${SCRATCH_INPUT}/data/tau_raw"

# Follow symlink source and copy actual files.
rsync -aL --info=stats2 "${SRC_DATA_DIR}/" "${SCRATCH_INPUT}/data/tau_raw/"
cp -f "${SRC_DEMO_CSV}" "${SCRATCH_INPUT}/demo.csv"

echo "[stage] scratch usage after copy:"
df -h /scratch/local || true
du -sh "${SCRATCH_INPUT}/data/tau_raw" || true

if [[ "${STAGE_ONLY:-0}" == "1" ]]; then
  echo "[stage] STAGE_ONLY=1 set, exiting after scratch staging test."
  exit 0
fi

# Multi-output regression on 4 regions
python run.py \
  --no-tune \
  --dataset Gothenburg \
  --data_type tau_raw \
  --input_path "${SCRATCH_INPUT}" \
  --epochs 4 \
  --batch_size 4 \
  --targets MetaTemporal,MesialTemporal,Frontal,TemporoParietal \
  --stratifycvby site,MetaTemporal,MesialTemporal,Frontal,TemporoParietal \
  --model_name_extra raw-multioutput-test
