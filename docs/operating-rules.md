# Operating Rules And Grievances

This is now a compatibility wrapper. The canonical rules live in:

- [reflections.md](reflections.md)
- [runbooks/README.md](runbooks/README.md)
- [code/04-metrics-and-video-contract.md](code/04-metrics-and-video-contract.md)
- [code/05-remote-ec2-contract.md](code/05-remote-ec2-contract.md)

The most important rules remain:

- reuse existing rollout/eval/plot/video code,
- do not silently change model architecture, horizon, optimizer, metrics, preprocessing, or reference actions,
- use `uv` for local Python,
- download artifacts and shut down EC2 after remote runs,
- commit coherent code/docs units.
