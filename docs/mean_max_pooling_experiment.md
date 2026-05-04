# Mean + Max Pooling Experiment

This experiment adds an alternative action training entrypoint that uses mean + max temporal pooling.

## Why

The default action model uses masked mean pooling over the bidirectional GRU outputs. Mean pooling is stable, but short and decisive temporal changes can be diluted. This matters for classes like `collapse`, where the onset of falling may be more important than the average pose over the whole sequence.

Mean + max pooling keeps both views:

- mean pooling: stable overall sequence behavior
- max pooling: strong localized temporal activations

## Files

- `app/action_training_pipeline_mean_max.py`
- `configs/experiments/action_training.aihub_shell.mean_max_pooling.json`

## How to run

From the project root:

```bash
python app/action_training_pipeline_mean_max.py --config configs/experiments/action_training.aihub_shell.mean_max_pooling.json --stage all
```

For train-only after manifests are prepared:

```bash
python app/action_training_pipeline_mean_max.py --config configs/experiments/action_training.aihub_shell.mean_max_pooling.json --stage train
```

## Important note

This entrypoint intentionally uses a separate workspace:

```text
../training_data/action_pipeline_aihub_mean_max
```

It also sets:

```json
"resume_from_best": false
```

That makes the comparison cleaner and prevents old mean-only classifier checkpoints from affecting the experiment.

## What to compare

After the run, compare against the default model:

1. Best validation macro F1
2. Validation accuracy
3. `loitering -> collapse` confusion count
4. `collapse` F1
5. `abduction` precision / recall / F1
6. Train/validation loss gap

Use:

```bash
python app/training_result_audit.py --workspace ../training_data/action_pipeline_aihub_mean_max
```

## Implementation detail

The new entrypoint monkey-patches `action_model.TemporalPoseClassifier` before calling the existing training pipeline. This keeps the existing high-risk pipeline file mostly untouched while allowing a controlled model architecture experiment.
