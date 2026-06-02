from pathlib import Path
import json
import re

root = Path.home() / "robotics/runs/libero-language-sweep-20260529-1120"
pids = ["P0", "P1", "P2", "P3", "C1", "C2", "C3", "C4", "O1", "O2"]


def count_success_text(paths):
    text = "\n".join(p.read_text(errors="ignore") for p in paths if p.exists())
    vals = re.findall(r"Success:\s*(True|False)", text)
    return vals.count("True"), len(vals)


rows = []
for pid in pids:
    s, n = count_success_text([root / "pi" / pid / "eval.log"])
    rows.append(("pi05_libero", pid, s, n))

for pid in pids:
    s, n = count_success_text([root / "molmo" / pid / "eval.log.inner"])
    rows.append(("molmoact2_libero", pid, s, n))

for pid in pids:
    paths = list((root / "cosmos" / pid / "logs").glob("*.txt"))
    s, n = count_success_text(paths)
    rows.append(("cosmos_policy", pid, s, n))

for pid in pids:
    paths = list((root / "ge-act" / pid / "libero_spatial").glob("*/inference_*.txt"))
    s, n = count_success_text(paths)
    rows.append(("ge_act", pid, s, n))

for pid in pids:
    result_paths = list((root / "fastwam" / pid / "output" / "libero_spatial").glob("*results.json"))
    if result_paths:
        data = json.loads(result_paths[0].read_text())
        text = json.dumps(data)
        match = re.search(r'"successes"\s*:\s*(\d+)', text)
        s = int(match.group(1)) if match else len(re.findall(r"true", text, re.I))
        n = 3
    else:
        text = "\n".join(
            p.read_text(errors="ignore")
            for p in [root / "fastwam" / pid / "eval.comma-repair.log", root / "fastwam" / pid / "eval.log"]
            if p.exists()
        )
        matches = re.findall(r"Task 0 completed:\s*(\d+)/(\d+) successes", text)
        if matches:
            s, n = map(int, matches[-1])
        else:
            s, n = 0, 0
    rows.append(("fastwam", pid, s, n))

out = root / "summary_success.tsv"
out.write_text(
    "model\tprompt_id\tsuccesses\tepisodes\tsuccess_rate\n"
    + "\n".join(f"{m}\t{p}\t{s}\t{n}\t{(s / n if n else 0):.3f}" for m, p, s, n in rows)
    + "\n"
)
print(out)
print(out.read_text())
