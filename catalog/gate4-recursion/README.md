# Gate 4: Continuous Recursion & The Compound Derivative (`catalog/gate4-recursion/`)

**Metric**: `Judgment-cost per verified catalog entry (wall-clock hours & human intervention ratio)`  
**Target**: Declining human intervention hours and sub-6 hour end-to-end RFC verification cycles  
**Primary Hardware Target**: AMD Radeon RX 7900 XTX (ROCm PJRT)  
**Architecture Guide**: [docs/architecture/rsi_gates.md](../../docs/architecture/rsi_gates.md)

---

## 1. Scope & Objective

Gate 4 evaluates whether the agent research apparatus operates in a **continuous, self-reinforcing recursive loop** ($G_n \to G_{n+1} \to \dots$):
- Measuring the **derivative of judgment**: does the wall-clock time from proposal to verified merge decrease across generations? Does the proportion of autonomous machine evaluation increase?
- Maintaining strict verification discipline: the model proposes, but the discrete host-side gate disposes and commits.
- Evaluating multi-turn stability over adversarial horizons to ensure zero invariant degradation.

## 2. Contained Experiment Pods (Generation $G_3$, E21–E23)

- [`e21-kb-write-path/`](e21-kb-write-path/): Schema-constrained KB write path; guaranteed **0 illegal role assignments or contradictory facts** commit to memory.
- [`e22-ingraph-dispatch/`](e22-ingraph-dispatch/): In-graph tool dispatch; achieved $100\%$ tool-token emission, and established mechanical sympathy ($42.20\ \mu\text{s}$ host discrete lookup vs $691.94\ \mu\text{s}$ device dispatch).
- [`e23-state-reduce/`](e23-state-reduce/): Verified KB accumulation over an adversarial horizon; guaranteed **0 RBAC invariant violations** committed under attack.
