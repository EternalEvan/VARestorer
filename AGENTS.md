# Repository Guidelines

## Project Structure & Module Organization
`train.py`, `trainer.py`, and the `train_*.py` variants drive experiments. `scripts/` contains the main shell entrypoints for training, inference, and benchmark runs. Core model, dataset, and distributed utilities live in `infinity/` (`models/`, `dataset/`, `utils/`). Evaluation code is under `evaluation/` with subfolders for GenEval, HPSv2, ImageReward, and validation loss. General helpers live in `utils/` and `compare_img/`. Treat `weights/`, `checkpoints/`, `local_output/`, `wandb/`, and visualization folders as generated artifacts, not source.

## Build, Test, and Development Commands
Use the repository root as the working directory.

- `pip3 install -r requirements.txt`: install base Python dependencies; the repo expects `torch==2.5.1`.
- `bash scripts/train.sh`: launch the default single-node training job and write logs/checkpoints under `local_output/` and `checkpoints/`.
- `bash scripts/infer.sh`: run prompt-based inference through `tools/run_infinity.py`.
- `bash scripts/eval.sh`: execute the configured benchmark pipeline; GenEval is enabled in the current script.
- `python3 evaluate.py`: compute IQA metrics for prepared prediction/reference folders.
- `python3 test_pyiqa.py` and `python3 test_timm.py`: quick dependency smoke tests.

## Coding Style & Naming Conventions
Python follows existing project conventions: 4-space indentation, `snake_case` for functions, variables, and file names, and `CamelCase` for classes. Keep new CLI flags and config names consistent with existing argument patterns such as `--local_out_path` and `--vae_ckpt`. There is no root formatter or linter config checked in, so match surrounding style and keep imports grouped cleanly.

## Testing Guidelines
There is no centralized `pytest` suite in the root project. Add focused smoke tests as `test_*.py` scripts near the repo root or beside the affected module. Prefer checks that verify imports, model loading, or a minimal forward pass. For training or evaluation changes, include the exact command used and the expected output location in your PR.

## Commit & Pull Request Guidelines
Recent commit subjects are short and plain, for example `final version` and `with metrics`. Keep that style concise, but make it more descriptive when possible, such as `add GenEval logging` or `fix VAE checkpoint path`. PRs should state the purpose, list the commands run, note any required weights or datasets, and include sample outputs or screenshots for inference or metric changes.

## Artifact & Configuration Hygiene
Do not commit model weights, checkpoints, `local_output/` runs, or machine-specific paths. The repo already ignores some generated files, but contributors should still review `git status` before pushing, especially after running training or evaluation scripts.
