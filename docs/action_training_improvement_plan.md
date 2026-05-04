# Action Training Improvement Plan

This note documents the follow-up changes for the action-classification training dashboard results.

## Current finding

The dashboard result showed a clear gap between training and validation behavior:

- Training loss continued to decrease.
- Validation loss stayed high and unstable.
- Best validation performance occurred before the latest epoch.
- The largest visible confusion was `loitering -> collapse`.
- `abduction` recall was not the worst problem, but `abduction` precision was low.

Because of this, the first goal is not to simply add more epochs. The first goal is to make evaluation easier and run fair comparison experiments.

## Added experiment configs

### `configs/experiments/action_training.aihub_shell.fresh_stability.json`

Use this for a fresh run with the current stability-oriented settings:

- Longer temporal window: `sequence_length=64`, `max_frames_to_scan=220`
- Slightly stricter pose-quality filtering
- Lower learning rate
- Stronger regularization
- Lower focal gamma
- Mild `abduction` multiplier
- `resume_from_best=false`

This config is best for checking whether the new settings actually help from a clean start.

### `configs/experiments/action_training.aihub_shell.cross_entropy_baseline.json`

Use this as a baseline against focal loss:

- Same temporal and regularization settings as the fresh stability config
- `loss=cross_entropy`
- `resume_from_best=false`

This helps answer whether focal loss is helping or making the noisy validation behavior worse.

## Added audit utility

### `app/training_result_audit.py`

Run it after training:

```bash
python app/training_result_audit.py --workspace ../training_data/action_pipeline_aihub
```

For a fresh experiment workspace:

```bash
python app/training_result_audit.py --workspace ../training_data/action_pipeline_aihub_fresh_stability
```

It summarizes:

- Best checkpoint vs latest epoch metrics
- Top confusion pairs
- Class-level raw-to-prepared retention
- Useful artifact paths such as `validation_error_analysis.json`, `false_negative_examples.json`, and `confusion_pair_examples.json`

## Recommended next checks

After each training run, compare:

1. Best validation macro F1
2. Final validation accuracy
3. `abduction` precision / recall / F1
4. `loitering -> collapse` confusion count
5. Train/validation loss gap
6. Class-level retention rate

## Code-level changes still recommended

These are larger changes and should be implemented in a separate PR after the experiment configs are tested:

1. Dashboard should display best checkpoint metrics and latest epoch metrics separately.
2. Dashboard should link or render `confusion_pair_examples.json` and false-negative examples.
3. The model can be improved by adding mean + max temporal pooling instead of mean-only pooling.
4. Train-only pose augmentation can be added for weak jitter, horizontal flip, and temporal dropout.
5. Fresh-train and resume-train controls can be separated in the dashboard UI.

These changes touch larger files and should be tested carefully because the current dashboard and training pipeline are already integrated with queueing, AIHub lookup, and Pages sync.
