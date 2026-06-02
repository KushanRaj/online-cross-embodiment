#!/bin/bash
# =============================================================
# Setup: ManiSkill 3 + robosuite + LIBERO on Ubuntu (RTX 3060 x2)
# =============================================================
# What this gives you:
#   - Everything from the Mac setup PLUS:
#   - ManiSkill 3 with GPU-accelerated physics (Vulkan + CUDA)
#   - 30-50+ manipulation tasks, 20+ robot embodiments
#   - GPU rendering for fast trajectory collection
#
# Requirements: Ubuntu 22.04+, NVIDIA driver 525+, CUDA 11.8+
# Time: ~30-45 minutes
# =============================================================

set -e

echo "=== Step 0: Check NVIDIA/Vulkan prerequisites ==="
echo "NVIDIA driver:"
nvidia-smi | head -4

echo ""
echo "Vulkan support:"
# If this fails, install: sudo apt install vulkan-tools mesa-vulkan-drivers
vulkaninfo --summary 2>/dev/null | head -10 || echo "WARNING: vulkaninfo not found. Install: sudo apt install vulkan-tools"

echo ""
echo "If Vulkan is not working, run these first:"
echo "  sudo apt update"
echo "  sudo apt install -y vulkan-tools mesa-vulkan-drivers libvulkan-dev"
echo "  # For headless rendering (no monitor):"
echo "  sudo apt install -y libegl1-mesa-dev libgles2-mesa-dev"
echo ""
read -p "Press Enter to continue (or Ctrl+C to fix Vulkan first)..."

echo "=== Step 1: Create conda environment ==="
conda create -n robosim python=3.10 -y
conda activate robosim

echo "=== Step 2: Install PyTorch (CUDA) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "=== Step 3: Install MuJoCo + robosuite + LIBERO ==="
pip install mujoco robosuite libero

# For headless rendering (no monitor attached)
export MUJOCO_GL=egl

echo "=== Step 4: Verify robosuite ==="
MUJOCO_GL=egl python -c "
import robosuite as suite
env = suite.make(
    env_name='Lift',
    robots='Panda',
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
)
obs = env.reset()
print('robosuite OK. Robots:', suite.ALL_ROBOTS)
env.close()
"

echo "=== Step 5: Verify LIBERO ==="
MUJOCO_GL=egl python -c "
from libero.libero.benchmark import get_benchmark
for suite_name in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']:
    b = get_benchmark(suite_name)
    print(f'{suite_name}: {b.n_tasks} tasks')
print('LIBERO OK!')
"

echo "=== Step 6: Install ManiSkill 3 ==="
pip install --upgrade mani_skill

echo "=== Step 7: Download ManiSkill assets ==="
# Download commonly used task assets
python -m mani_skill.utils.download_asset "PickCube-v1"
python -m mani_skill.utils.download_asset "StackCube-v1"
python -m mani_skill.utils.download_asset "PegInsertionSide-v1"
python -m mani_skill.utils.download_asset "PickSingleYCB-v1"

echo "=== Step 8: Verify ManiSkill 3 ==="
python -c "
import mani_skill.envs
import gymnasium as gym

# Test basic environment creation
env = gym.make(
    'PickCube-v1',
    obs_mode='rgbd',
    control_mode='pd_ee_delta_pos',
    render_mode='rgb_array',
    num_envs=1,
)
obs, info = env.reset()
print('ManiSkill OK!')
print('Observation keys:', obs.keys() if hasattr(obs, 'keys') else type(obs))
print('Action space:', env.action_space)
env.close()
"

echo "=== Step 9: Install LeRobot ==="
pip install lerobot

echo "=== Step 10: Test with different robots in robosuite ==="
MUJOCO_GL=egl python -c "
import robosuite as suite

# Test multiple embodiments
for robot in ['Panda', 'Sawyer', 'IIWA', 'Jaco', 'UR5e']:
    env = suite.make(
        env_name='Lift',
        robots=robot,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
    )
    obs = env.reset()
    action_dim = env.action_spec[0].shape[0]
    print(f'{robot}: action_dim={action_dim}, obs_keys={len(obs.keys())}')
    env.close()
print('All robots working!')
"

echo ""
echo "=== DONE ==="
echo ""
echo "What you now have:"
echo "  - robosuite: Panda, Sawyer, IIWA, Jaco, Kinova3, UR5e, Baxter"
echo "  - LIBERO: 130 tasks (4 suites)"
echo "  - ManiSkill 3: PickCube, StackCube, PegInsertion, PickSingleYCB"
echo "  - LeRobot: ready for VLA checkpoints"
echo ""
echo "GPU memory note (2x RTX 3060 12GB):"
echo "  - Sim environments: ~1-2GB VRAM"
echo "  - SmolVLA inference: ~2-4GB VRAM (FITS on one 3060)"
echo "  - OpenVLA-7B inference: ~16GB (needs quantization or AWS)"
echo "  - pi0/pi0.5 inference: ~8-12GB (might fit on one 3060 with fp16)"
echo ""
echo "For headless operation, always set: export MUJOCO_GL=egl"
