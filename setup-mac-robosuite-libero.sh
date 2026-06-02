#!/bin/bash
# =============================================================
# Setup: robosuite + LIBERO on macOS (Apple Silicon)
# =============================================================
# What this gives you:
#   - MuJoCo physics (bundled with robosuite)
#   - robosuite: 8 robot embodiments, custom task creation
#   - LIBERO: 130 Franka manipulation tasks, 4 benchmark suites
#   - LeRobot: framework for loading VLA checkpoints later
#
# Requirements: Python 3.10+, Homebrew
# Time: ~15-20 minutes
# =============================================================

set -e

echo "=== Step 1: Create conda environment ==="
# If you don't have conda, install miniforge first:
#   brew install miniforge
#   conda init zsh

conda create -n robosim python=3.10 -y
conda activate robosim

echo "=== Step 2: Install MuJoCo + robosuite ==="
# MuJoCo is now bundled with the mujoco pip package (no separate download needed)
pip install mujoco
pip install robosuite

echo "=== Step 3: Verify robosuite ==="
python -c "
import robosuite as suite

# List available robots
print('Available robots:', suite.ALL_ROBOTS)
# Should show: ['Panda', 'Sawyer', 'IIWA', 'Jaco', 'Kinova3', 'UR5e', 'Baxter']

# List available environments
print('Available environments:', suite.ALL_ENVIRONMENTS)

# Quick test: create a Panda Lift environment
env = suite.make(
    env_name='Lift',
    robots='Panda',
    has_renderer=False,       # Set True if you want to see it (needs display)
    has_offscreen_renderer=True,
    use_camera_obs=True,
    reward_shaping=True,
)
obs = env.reset()
print('Observation keys:', list(obs.keys()))
print('Action space dimension:', env.action_spec[0].shape)
action = env.action_spec[0]  # low bound
obs, reward, done, info = env.step(action)
print('Step succeeded! Reward:', reward)
env.close()
print('robosuite is working!')
"

echo "=== Step 4: Install LIBERO ==="
# LIBERO installs on top of robosuite
pip install libero

# Download LIBERO benchmark datasets
python -c "
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
import os

# This downloads task definitions (not demo data - that's separate)
for suite_name in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']:
    benchmark = get_benchmark(suite_name)
    print(f'{suite_name}: {benchmark.n_tasks} tasks')
    # Print first 3 task names
    for i in range(min(3, benchmark.n_tasks)):
        print(f'  Task {i}: {benchmark.get_task(i).name}')
print('LIBERO benchmark loaded!')
"

echo "=== Step 5: Install LeRobot (for VLA checkpoints later) ==="
pip install lerobot

echo "=== Step 6: Quick visual test (optional - needs display) ==="
echo "Run this manually if you want to see the sim:"
echo ""
echo "python -c \""
echo "import robosuite as suite"
echo "env = suite.make("
echo "    env_name='Lift',"
echo "    robots='Panda',"
echo "    has_renderer=True,"
echo "    render_camera='frontview',"
echo ")"
echo "env.reset()"
echo "for _ in range(100):"
echo "    action = env.action_spec[0]"
echo "    env.step(action)"
echo "    env.render()"
echo "env.close()"
echo "\""

echo ""
echo "=== DONE ==="
echo ""
echo "What you now have:"
echo "  - robosuite with 7 robots: Panda, Sawyer, IIWA, Jaco, Kinova3, UR5e, Baxter"
echo "  - LIBERO with 130 tasks across 4 suites"
echo "  - LeRobot for loading VLA checkpoints"
echo ""
echo "Next: To run VLA models, use your Ubuntu desktop or AWS (needs CUDA)."
echo "       The Mac setup is for exploring tasks, visualizing, and lightweight dev."
