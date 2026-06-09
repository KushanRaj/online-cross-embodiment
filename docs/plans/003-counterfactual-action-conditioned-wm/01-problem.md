# 01 - Problem

Current Phase 3 asks:

```text
Cosmos imagines P from C.
Policy takes action a.
Does IDM(C,P) match a?
```

This is useful, but it is not the cleanest causal question.

The stronger question is:

```text
Given current C and arbitrary action chunk a,
what future P(a) does the world model predict?
Does P(a) match observed R after executing a?
```

That requires a genuinely action-conditioned world model.
