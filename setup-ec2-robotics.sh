#!/usr/bin/env bash
# =============================================================
# EC2 setup: MuJoCo + robosuite + LIBERO + ManiSkill + LeRobot
# =============================================================
# Intended for Ubuntu 22.04/24.04 GPU instances. Safe to run via
# EC2 user-data or manually over SSH.
#
# Usage:
#   bash setup-ec2-robotics.sh
#
# Optional environment variables:
#   CONDA_ENV=robosim
#   INSTALL_MANISKILL=1
#   CHECKPOINT_DIR=/mnt/checkpoints
# =============================================================

set -euo pipefail

LOG_DIR="${LOG_DIR:-$HOME/setup-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/setup-ec2-robotics-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

CONDA_ENV="${CONDA_ENV:-robosim}"
INSTALL_MANISKILL="${INSTALL_MANISKILL:-1}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"

echo "=== EC2 robotics setup started: $(date) ==="
echo "Log: $LOG_FILE"
echo "Conda env: $CONDA_ENV"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== NVIDIA GPU ==="
  nvidia-smi
else
  echo "WARNING: nvidia-smi not found. Use a GPU AMI or install NVIDIA drivers before GPU workloads."
fi

echo "=== System packages ==="
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  ca-certificates \
  curl \
  ffmpeg \
  git \
  git-lfs \
  libegl1 \
  libegl1-mesa-dev \
  libgl1 \
  libgl1-mesa-dev \
  libgles2-mesa-dev \
  libglfw3 \
  libglfw3-dev \
  libosmesa6-dev \
  libvulkan-dev \
  mesa-vulkan-drivers \
  unzip \
  vim \
  vulkan-tools \
  wget

git lfs install || true

echo "=== Miniforge ==="
if [ ! -x "$MINIFORGE_DIR/bin/conda" ]; then
  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64) MINIFORGE_ARCH="x86_64" ;;
    aarch64|arm64) MINIFORGE_ARCH="aarch64" ;;
    *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
  esac
  curl -L -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${MINIFORGE_ARCH}.sh"
  bash /tmp/miniforge.sh -b -p "$MINIFORGE_DIR"
fi

# shellcheck source=/dev/null
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda config --set auto_activate_base false

echo "=== Conda env ==="
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda create -n "$CONDA_ENV" python=3.10 -y
fi
conda activate "$CONDA_ENV"

python -m pip install --upgrade pip setuptools wheel

echo "=== Python GPU stack ==="
python -m pip install --upgrade \
  torch \
  torchvision \
  torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

echo "=== Robotics packages ==="
python -m pip install --upgrade \
  "huggingface_hub[cli]" \
  accelerate \
  datasets \
  gymnasium \
  h5py \
  imageio \
  imageio-ffmpeg \
  ipykernel \
  jupyterlab \
  matplotlib \
  mujoco \
  numpy \
  opencv-python \
  pandas \
  pillow \
  scipy \
  scikit-learn \
  tqdm \
  transformers

# Keep robosuite and LIBERO close to the versions used by the benchmark stack.
python -m pip install "robosuite==1.4.1" libero

echo "=== LeRobot ==="
python -m pip install "lerobot[smolvla]"

if [ "$INSTALL_MANISKILL" = "1" ]; then
  echo "=== ManiSkill ==="
  python -m pip install --upgrade mani_skill
fi

echo "=== Persistent env vars ==="
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/robotics-env.sh" <<'EOF'
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
EOF

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

echo "=== Smoke test: imports ==="
python - <<'PY'
import importlib
mods = ["torch", "mujoco", "robosuite", "libero", "gymnasium", "lerobot"]
for mod in mods:
    m = importlib.import_module(mod)
    print(f"{mod}: OK {getattr(m, '__version__', '')}")
PY

echo "=== Smoke test: robosuite Lift ==="
python - <<'PY'
import robosuite as suite
env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    reward_shaping=True,
)
obs = env.reset()
print("robosuite robots:", suite.ALL_ROBOTS)
print("obs keys:", list(obs.keys())[:8])
print("action dim:", env.action_spec[0].shape)
env.close()
PY

echo "=== Smoke test: LIBERO benchmark metadata ==="
python - <<'PY'
from libero.libero.benchmark import get_benchmark
for suite_name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    benchmark = get_benchmark(suite_name)
    print(f"{suite_name}: {benchmark.n_tasks} tasks")
PY

if [ "$INSTALL_MANISKILL" = "1" ]; then
  echo "=== Smoke test: ManiSkill import ==="
  python - <<'PY'
import gymnasium as gym
import mani_skill.envs
env = gym.make(
    "PickCube-v1",
    obs_mode="rgbd",
    control_mode="pd_ee_delta_pos",
    render_mode="rgb_array",
    num_envs=1,
)
obs, info = env.reset()
print("ManiSkill PickCube OK")
print("action space:", env.action_space)
env.close()
PY
fi

echo "=== Done ==="
echo "Activate with:"
echo "  source $MINIFORGE_DIR/etc/profile.d/conda.sh"
echo "  conda activate $CONDA_ENV"
echo ""
echo "Checkpoint download is intentionally separate."
echo "Example:"
echo "  CHECKPOINT_DIR=\${CHECKPOINT_DIR:-$HOME/vla-checkpoints} bash download-all-checkpoints.sh --dry-run"
echo "Finished: $(date)"
