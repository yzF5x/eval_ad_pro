# eval_ad_pro_refactor

Refactored project layout aligned with `eval_ad_perhead` style:

- `configs/`: prompt configuration + YAML config loader
- `models/`: model handlers and factory (`qwen` / `internvl` / `glm`)
- `utils/`: utilities (metrics, loader, visual tools, monkey patch, preprocessing)
- `generator.py`: generation + attention saving entrypoint
- `evaluator.py`: anomaly metrics + saved-attention evaluation entrypoint

## Config-first usage

All business parameters are loaded from YAML config with three sections:

1. `shared`: parameters shared by both generator and evaluator
2. `generator`: generator-only parameters
3. `evaluator`: evaluator-only parameters

`dataset_path` / `save_dir` / `generated_dir` / `replace_path` are auto-filled from
`configs/dataset_config.py` by the CLI argument `--dataset` (e.g. `wfdd`, `mvtec`).

Example configs are provided in `configs/yaml/`:

- `configs/yaml/config_qwen.yaml`
- `configs/yaml/config_internvl.yaml`
- `configs/yaml/config_glm.yaml`

Install YAML dependency:

```bash
pip install pyyaml
```

Run generator:

```bash
python generator.py --config configs/yaml/config_qwen.yaml --dataset wfdd
```

Run evaluator:

```bash
python evaluator.py --config configs/yaml/config_qwen.yaml --dataset wfdd
```

## Scripts

PowerShell scripts are in `scripts/`:

- `scripts/run_generator.ps1`
- `scripts/run_evaluator.ps1`
- `scripts/run_pipeline.ps1`

Run from `eval_ad_pro_refactor`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_generator.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_evaluator.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_pipeline.ps1
```

Override config and dataset:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_pipeline.ps1 -Config "configs/yaml/config_glm.yaml" -Dataset Liver_AD
```
