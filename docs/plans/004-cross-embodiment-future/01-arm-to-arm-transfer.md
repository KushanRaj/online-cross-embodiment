# 01 - Arm-To-Arm Transfer

The future hypothesis:

```text
The high-level model may reason well,
but low-level actuator conditioning is embodiment-specific.
```

Start with arm-to-arm transfer, not humanoid/mobile manipulation.

Possible shared output:

```text
end-effector delta pose + gripper
```

The low-level controller for each arm maps that to joints.

This avoids direct Franka-to-UR5/Kinova joint mapping, which is not clean because joint counts, limits, kinematics, workspaces, and controllers differ.
