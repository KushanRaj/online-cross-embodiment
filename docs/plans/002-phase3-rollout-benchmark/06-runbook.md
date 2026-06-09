# 06 - Runbook

Use the operational runbook:

- [../../runbooks/phase3-evaluate-and-render.md](../../runbooks/phase3-evaluate-and-render.md)
- [../../runbooks/ec2-remote-run.md](../../runbooks/ec2-remote-run.md)

## Before Running

State:

```text
environment
policy
world model/checkpoint
IDM checkpoint/type
horizon
reference action dataset
task list
output path
shutdown plan
```

## After Running

Produce:

```text
summary.json
episodes.csv
deviation summary
success/failure aggregate plots
prediction metric videos
local downloaded artifact folder
```

## Do Not

- write a new renderer,
- change metric definitions,
- run one-step IDM against a 16-step future,
- evaluate IDM-as-policy only against its own executed action,
- leave EC2 running.
