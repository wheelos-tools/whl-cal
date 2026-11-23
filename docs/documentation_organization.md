# Documentation ownership

Use one location for each kind of information so the maintained docs, agent
guidance, and historical evidence do not become competing sources of truth.

## Locations

- `docs/`: maintained project documentation—supported workflows, user-facing
  runbooks, design contracts, and result interpretation. Keep commands,
  defaults, and output names aligned with the implementation.
- `.agents/knowledge/`: concise, durable engineering facts and decision
  summaries for agents and maintainers. Link to supporting docs or notes
  instead of copying long explanations.
- `.agents/skills/`: repeatable procedures for carrying out a class of work.
- `.agents/notes/`: dated investigations, experiment records, historical
  runbooks, and temporary follow-ups. A note may describe an older state; check
  the current code and maintained docs before treating it as current.
- `.github/`: repository governance and automation.

The former top-level `context/` collection has been moved into
`.agents/knowledge/` or `.agents/notes/` according to its role. Do not recreate
a parallel context tree.

## Updating documentation

When behavior, a CLI interface, a default, or an output contract changes:

1. Update the relevant maintained document in `docs/`.
2. Update concise agent knowledge only when the change affects a durable
   invariant or engineering decision; link to the maintained doc or evidence.
3. Put experiment details in `.agents/notes/` and label them with the data/run
   identity and date.
4. Validate documented commands and file paths against the implementation.
   Run the smallest relevant smoke check before changing any `last_tested`
   metadata.

If documents conflict, implementation is authoritative. Record unresolved
uncertainty explicitly rather than preserving conflicting claims.
