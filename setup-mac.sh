#!/bin/bash
# =============================================================
# SETUP: robosuite + LIBERO on macOS Apple Silicon
# =============================================================
# Run with: source setup-mac.sh
# (must use 'source' not './setup-mac.sh' so conda activate works)
# =============================================================

set -e

eval "$(conda shell.bash hook)"

# --- Environment 1: robosuite (latest, explore all robots) ---
echo "=== Creating robosuite environment ==="
conda create -n robosuite python=3.10 -y
conda activate robosuite

pip install torch torchvision
pip install mujoco robosuite

echo "=== Test robosuite ==="
python -c "
import robosuite as suite
print('Robots:', suite.ALL_ROBOTS)
print('Envs:', suite.ALL_ENVIRONMENTS)
env = suite.make('Lift', robots='Panda', has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
obs = env.reset()
print('Action dim:', env.action_spec[0].shape)
env.close()
print('robosuite OK!')
"

# --- Environment 2: LIBERO (needs robosuite 1.4.0) ---
echo "=== Creating LIBERO environment ==="
conda create -n libero python=3.10 -y
conda activate libero

pip install torch torchvision
pip install robosuite==1.4.0
pip install libero

echo "=== Test LIBERO ==="
python -c "
from libero.libero.benchmark import get_benchmark
for s in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']:
    b = get_benchmark(s)
    print(f'{s}: {b.n_tasks} tasks')
print('LIBERO OK!')
"

# Switch back to robosuite env
conda activate robosuite

echo ""
echo "=== DONE ==="
echo "  conda activate robosuite  # explore robots, custom tasks"
echo "  conda activate libero     # LIBERO 130-task benchmark"
