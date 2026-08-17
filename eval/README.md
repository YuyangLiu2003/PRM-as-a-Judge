# Eval Toolkit

This folder contains the manifest-first PRM-as-a-Judge v1.5 evaluation toolkit. Manifests may be UTF-8 with or without a BOM.

## Verified setup and public model

The base toolkit supports Python 3.10 or newer. The released Dopamine constraints are verified on Linux x86_64, Python 3.10, and CUDA 12.8:

```bash
conda create -n prm-as-a-judge python=3.10 -y
conda activate prm-as-a-judge
python -m pip install -e ".[dopamine]" \
  -c constraints/dopamine-cu128-py310.txt
python -m pip check

hf download tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview \
  --revision 980f3d1819c870f62c36169a9486e971049bb09a \
  --local-dir PRM/Robo-Dopamine-GRM-2.0-8B-Preview
```

The command follows the official [`hf download` guide](https://huggingface.co/docs/huggingface_hub/guides/download).

The Preview directory above is the default. To use a compatible Pro checkpoint, set `PRM_PATH` explicitly:

```bash
MANIFEST=/path/to/cases.jsonl \
PRM_PATH=/path/to/Robo-Dopamine-GRM-8B-Pro \
VISUALIZE=1 \
bash eval/run_eval.sh
```

Without `PRM_PATH`, the bundled example command uses the public Preview default and `EVAL_MODE=incremental`:

```bash
MANIFEST=eval/examples/manifest_demo_cases.jsonl VISUALIZE=1 bash eval/run_eval.sh
```

## Manifest contract

The bundled demo stores its manifest and assets directly in `eval/examples/`:

```text
eval/examples/manifest_demo_cases.jsonl
eval/examples/{task}.jpg
eval/examples/{task}_ep{n}_{high|wrist}.mp4
```

The canonical video fields are:

- `video` (required): main, third-person, or overhead view.
- `wrist_video` (optional): the wrist-camera view of a single-arm robot. Dopamine duplicates it into the two wrist input slots required by the backend.
- `left_wrist_video` and `right_wrist_video` (optional): the left and right wrist views of a bimanual robot; they must be provided together.

Use `wrist_video` for a single-arm robot or the left/right pair for a bimanual robot, not both. Paths may be absolute or relative to the manifest. The bundled demo contains single-arm trajectories and therefore uses `video` plus `wrist_video`. Legacy `videos` objects with `high`, `left`, and `right` remain readable for compatibility but are no longer the recommended format. Fixed `videos/tasks/goals` directory discovery is no longer supported.

RoboMeter uses the canonical `video` view by default; select another manifest field with `--robometer-view`. Install it with `python -m pip install -e ".[robometer,visualize]"`.

## Recorded smoke and exit status

The lightweight base editable install includes Excel reporting support and can exercise metrics and reporting with two synthetic progress traces backed by real bundled MP4 files:

```bash
python -m pip install -e .
prm-judge eval \
  --manifest eval/examples/manifest_recorded.jsonl \
  --prm recorded \
  --visualize
```

After writing all available artifacts, the CLI exits 1 if any case has `status=error`. `--allow-partial` preserves error records but changes that final exit status to 0. Successful runs and valid dry-runs exit 0. In multi-GPU runs, workers write shards and the main process applies this rule after merging them; a worker crash still fails immediately.

## Outputs

Outputs are written to `eval/results/run_YYMMDD_HHMMSS/`:

- `run_params.json`
- `discovery_manifest.json`
- `per_case.jsonl`
- `metrics.xlsx` with `Cases`, `Task Summary`, `Model Summary Mean`, and `Model Summary Median` sheets
- `run_summary.json`
- `report.md`
- per-case `result_summary.json`
- `visualizations/report.html`
- `visualizations/curve_metrics.csv`
- `visualizations/visualization_report.md`
- `visualizations/case_plots.csv` and `visualizations/cases/*.png` when Matplotlib is installed
- run-level `robometer_server_PORT.log` when Robometer auto-start is used

In multi-GPU runs, worker-level process files are grouped under `shards/` while final outputs remain at the run root:

```text
run_YYMMDD_HHMMSS/
├── run_params.json
├── discovery_manifest.json
├── per_case.jsonl
├── metrics.xlsx
├── run_summary.json
├── report.md
├── shards/
│   ├── shard_000/
│   │   ├── run_params.json
│   │   ├── discovery_manifest.json
│   │   └── per_case.jsonl
│   └── shard_001/
│       ├── run_params.json
│       ├── discovery_manifest.json
│       └── per_case.jsonl
└── <benchmark>/<model>/<task>/<case_id>/
    ├── pred_vllm.json
    └── result_summary.json
```

Each worker manifest and JSONL contains only that worker's assigned cases. Per-case directories do not depend on the GPU count, and the main process merges all shard JSONL files into the canonical run-level `per_case.jsonl` before writing reports.

The report metric order is `M25`, `M50`, `M75`, `SR`, `MP`, `PPL`, `CRA`, `STR`, `DRR`, `FNS`, and `SQS`. Success-conditioned metrics use `MP >= 0.99` by default. Pass `--success-source label` to use manifest labels instead.

Task- and model-level `DRR` average only trajectories whose processed progress falls more than `1e-12` below a previous peak. The tolerance excludes floating-point tail noise, and per-case report cells are blank when no qualifying regression occurs. `drr_epsilon` remains only a numerical stabilizer in the recovery ratio.

## Interactive report

Use the built-in server for MP4 Range requests and reliable seeking:

```bash
prm-judge serve \
  --run-root eval/results/run_YYMMDD_HHMMSS \
  --host 127.0.0.1 \
  --port 8000
```

Open `http://127.0.0.1:8000/report.html`. For remote evaluation, tunnel the same loopback port:

```bash
ssh -L 8000:127.0.0.1:8000 user@server
```

ECharts 5.5.1 is embedded in the HTML, so charts need no network. Videos are not copied or embedded. Direct file opening works while the recorded paths remain accessible, but copying only `report.html` does not. Do not use `python -m http.server` for report playback because it does not reliably support the byte-range requests needed for seeking. Missing video files still produce a media error.

See the [online user guide](https://prm-as-a-judge.github.io/doc.html) for the full user guide.
