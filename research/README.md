# Research Documents

Thesis-facing research writeups, organized by the five project phases. Each phase folder is
self-contained: every document lives alongside the figures it references, so a folder can be
copied, reviewed, or archived as one unit.

| Folder | Phase | Scope |
|---|---|---|
| [phase1_data_baselines/](phase1_data_baselines/) | 1 | Market + news data collection, preprocessing, dataset construction, baseline model comparison |
| [phase2_cmtf_fusion/](phase2_cmtf_fusion/) | 2 | Cross-Modal Temporal Fusion (CMTF) model design and comparison against no-news / early-fusion / late-fusion baselines |
| [phase3_ablation_studies/](phase3_ablation_studies/) | 3 | Component-level ablations isolating each CMTF design choice |
| [phase4_multiagent_system/](phase4_multiagent_system/) | 4 | LangGraph multi-agent system built on the Phase 1–3 artifacts, evaluated against a base-LLM-only baseline |
| [phase5_realtime_chatbot/](phase5_realtime_chatbot/) | 5 | Real-time interactive chatbot deployment of the multi-agent system |

## Conventions

- One document = one folder = one topic. Name documents `NN_topic.md` (numeric prefix controls
  reading order within a phase).
- Figures referenced by a document live in the same folder as that document and are linked with
  relative markdown image syntax (`![caption](figure.png)`).
- Numeric claims in these documents should be reproducible from the code/results already checked
  into the repository — cite the source file or results path that produced them.
- Existing supporting material (`docs/reference/CMTF_FUSION_FINDINGS.md`, `RESULTS_IMPROVEMENT_LEVERS.md`,
  `MULTIAGENT_SYSTEM.md`, etc.) is not duplicated here; these documents cite it instead of
  re-deriving it.
