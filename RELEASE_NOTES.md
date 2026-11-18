# Release preparation

## Preserved research behaviour

Feature extraction, numerical thresholds, rule directions, active-rule selection, graph topology and the five-match screening decision are unchanged. No participant-level screening evaluation was rerun during cleanup. Published screening figures remain historical development-sample results, not independent validation.

## Changes

- Synthetic history is disabled by default in the UI and evaluation; explicit demo options remain available.
- The automated checklist is labelled as such. Its v2 scoring gives no context credit for synthetic history and does not assume a successful faithfulness check when missing. This is still not human evaluation.
- Ablation defaults to the current aggregate cutoff of five. Historical eleven-rule analyses require an explicit cutoff of six.
- Historical performance metadata in reports is identified as historical and not recomputed for the current recording. A different requested cutoff is explicitly distinguished from that record.
- Local labels can be configured with `--label-file`, `data/private/labels/` or `DAIC_WOZ_LABEL_FILES`.
- The live graph describes the jitter proxy accurately. Four policy edges have their unrelated or insufficient source labels removed and are explicitly marked design/policy. This is a targeted correction, not a full clinical evidence audit.
- Retrieval v1 is archived with its original graph and labels. The release corpus is versioned as v2 and reevaluated; graph metrics are unchanged, while TF-IDF metrics differ because its input text changed. Do not substitute v2 results for the thesis's original experiment without explaining this change.
- Browser-test configuration uses a configurable local URL and portable temporary directory.

## Local records and Git

Generated reports, participant-level results, uploaded sidecars, formative-review drafts, editor settings and tracked bytecode are removed from the Git index, not from disk. An additional local copy and SHA-256 inventory are kept in the ignored `docs/private_release_archive/` directory. The incomplete formative review remains incomplete; no ratings were invented.

Public `data/published_results/` contains an aggregate-only historical screening summary. The live graph and retrieval benchmark remain tracked. Historical source annotations in the archived retrieval graph are retained only to reproduce that experiment.

**Before making the repository public:** removal in a new commit does not erase previous commits. Check whether earlier participant data was pushed, review redistribution permissions and decide separately whether Git history needs to be rewritten. This cleanup does not alter history, contact hosting services or publish anything.

No clinical deployment claim is made. Measurement compatibility, missing-data handling, independent validation and a full claim-level graph evidence audit remain open research work.
