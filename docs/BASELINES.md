# Baseline adaptations

These implementations adapt external method mechanisms to a shared dual-G1
skill/observation interface. They are **not native reproductions of the original
benchmarks**. A method label, unit test or smoke output does not prove a
physical rollout occurred.

## Included comparisons

| Condition | Main distinction |
| --- | --- |
| Full | Hierarchical scoped repair plus verified strategy/action experience |
| Without dynamic experience | Disables strategy/exemplar retrieval and promotion, not safety or accounting |
| Flat reflective repair | Direct reflection and program revision without the hierarchical repair controller |
| REMAC-Adapted | Pre/post-condition reflection and bounded task replanning |
| COHERENT-Adapted | Central proposal, paired executor feasibility and feedback adjustment |
| CaP-Agent0-Adapted | Visual differencing, nine coding candidates and synthesis |
| ASPIRE-Adapted online | Bounded diagnosis/fix loop, executed candidate search and run-local findings |

The REMAC/COHERENT adapters under `src/coevolve_experiments/methods/` expose
decision-structure **contract harnesses**. The package's deterministic smoke
engine is not the native physical runner. Production prompt bundles and
historical runtime wrappers are separate and the latter are not distributed;
generic live integration remains fail-closed.

The README's selected results include heterogeneous runtime revisions and
continuations. Full reuses a historical cohort. They do not establish a
runtime-matched causal ablation or the performance of the original external
methods.

## CaP-Agent0

The adaptation follows [CaP-X](https://arxiv.org/abs/2603.22435) and its
[reference implementation](https://github.com/capgym/cap-x/tree/53e9966d7a8e2fa7494676772bccc35280f5c0ed).
Implementation: `coevolve_experiments.methods.cap_agent0.CaPAgent0Adapter`.

- Initial-scene analysis and subsequent before/after visual differencing supply
  actual images to candidate generation.
- Nine same-model candidates use temperatures 0.1–0.9, followed by synthesis
  at 0.2. Worker count is configurable from 1 to 9 without changing K.
- Five interface-compatible mathematical helpers are frozen before evaluation:
  vector normalization, segment interpolation, point transformation, matrix to
  quaternion conversion, and transform decomposition. This is a five-of-nine
  subset of the upstream prepared library, not the original robot API.
- Only the synthesized program reaches shared admission. Candidate generation
  alone consumes no physics; failed/interrupted model requests still count.
- The shared executor adds accepted-checkpoint rollback and bounded atomic
  team transactions. This differs from the upstream no-reset trial protocol.
  No CoEvolve strategy bank or another seed's successful trajectory is imported.

Callbacks must send image content, not just filenames in a text prompt, and be
thread-safe for concurrent requests. Preserve candidate order, feedback,
request receipts and resume state. The pure compiler
`coevolve_experiments.methods.team_program` lowers permitted helpers to literals;
it does not physically execute generated Python. SDK, team and physics checks
still apply.

## ASPIRE online adaptation

Mechanisms follow [ASPIRE](https://arxiv.org/abs/2607.00272) and the inspected
[reference revision](https://github.com/NVlabs/ASPIRE/tree/f4c8939aab0af9b97690c561bd80e282940f7886).
Implementation: `coevolve_experiments.methods.aspire.AspireAdapted`.

At each accepted checkpoint, the adapter inspects current observations, proposes
a program, and diagnoses failed executions using their traces and images.
After three fix candidates without progress, it can generate eight hypotheses
per search generation, using up to three physically evaluated same-checkpoint
parents and at most five generations. The shared run budget and terminal gates
can stop it earlier.

Candidate siblings execute serially from a certified common checkpoint.
Acceptance advances the state and ends that local comparison; unexecuted
siblings are discarded, not counted as failures. A dispatched unresolved
candidate remains charged and blocks ambiguous continuation.

Unlike the original development-seed/held-out workflow, this is an online
single-run adaptation: no extra development rollouts or held-out validation.
All model roles in the selected Table 2 adaptation use GPT-5.4-mini rather than
the upstream reference agent. Memory starts empty per seed; promoted findings
retain exact accepted executed code and evidence, not untested generalized
programs. Their applicability remains a hypothesis.

`AspireAdapted(run_dir, invoke)` alternates `next_candidate(context)`,
`record_dispatch(...)`, `record_result(..., defer_promotion=True)` and
`drain_pending_promotions()`. Drain only after native physical accounting is
committed. Repeated receipt imports are idempotent; conflicting transaction
identities are errors. State is persisted in `aspire_state.json`.

## Evaluation and attribution

Shared admission does not make every runtime revision identical. Record exact
source/provider identities, physical budgets, endpoint predicates and
continuation ancestry. For the base task, placement alone is not the full
retreat endpoint; additional-layout cases use placement and release.

Use the exact historical frozen runtime for historical audits. The current
contract adapters cannot substitute for missing physical evidence. Unit
tests with injected model responses test control flow, not provider availability,
successful robot behavior or transfer gains.

Keep external method names marked “Adapted” and cite their original work.
Algorithm attribution does not replace code/asset redistribution permissions;
third-party notices and project licensing require separate confirmation.
