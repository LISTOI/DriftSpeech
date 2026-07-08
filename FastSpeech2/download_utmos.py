import argparse
import os
from pathlib import Path

import torch


DEFAULT_REPO = "tarepan/SpeechMOS"
DEFAULT_MODEL = "utmos22_strong"
DEFAULT_OUTPUT_DIR = "offline_utmos"


def repo_cache_prefix(repo):
    return repo.replace("/", "_")


def find_hub_repo_dir(output_dir, repo):
    hub_dir = Path(output_dir) / "hub"
    prefix = repo_cache_prefix(repo)
    if not hub_dir.exists():
        raise FileNotFoundError("torch.hub directory not found: {}".format(hub_dir))
    candidates = [path for path in hub_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)]
    candidates = [path for path in candidates if (path / "hubconf.py").exists()]
    if not candidates:
        raise FileNotFoundError("No local hub repo for {} found under {}".format(repo, hub_dir))
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def ensure_output_dir(path, force):
    output_dir = Path(path)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise SystemExit("Output directory is not empty; pass --force to reuse it: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def readme_text(output_dir, repo_dir, model):
    return """# Offline UTMOS bundle

Copy this directory to the offline machine, then run:

```bash
export TORCH_HOME={output_dir}
```

Use this argument with `evaluate_flow_mel_utmos.py`:

```bash
--utmos_model_dir {repo_dir}
```

The model was validated locally with:

```python
torch.hub.load({repo_dir!r}, {model!r}, source="local")
```
""".format(output_dir=output_dir, repo_dir=repo_dir, model=model)


def write_readme(output_dir, repo_dir, model):
    path = Path(output_dir) / "README_offline_utmos.txt"
    path.write_text(readme_text(output_dir, repo_dir, model), encoding="utf-8")
    return path


def download_utmos(args):
    output_dir = ensure_output_dir(args.output_dir, args.force)
    os.environ["TORCH_HOME"] = str(output_dir.resolve())
    print("TORCH_HOME={}".format(os.environ["TORCH_HOME"]))
    print("Downloading/loading {}:{}".format(args.repo, args.model))
    model = torch.hub.load(args.repo, args.model)
    if hasattr(model, "eval"):
        model.eval()
    repo_dir = find_hub_repo_dir(output_dir, args.repo)
    print("Validating local load from {}".format(repo_dir))
    local_model = torch.hub.load(str(repo_dir), args.model, source="local")
    if hasattr(local_model, "eval"):
        local_model.eval()
    readme_path = write_readme(str(output_dir.resolve()), str(repo_dir.resolve()), args.model)
    print("Wrote {}".format(readme_path))
    print("Use this on the offline machine:")
    print("  export TORCH_HOME={}".format(str(output_dir.resolve())))
    print("  --utmos_model_dir {}".format(str(repo_dir.resolve())))
    return repo_dir


def self_test():
    assert repo_cache_prefix("tarepan/SpeechMOS") == "tarepan_SpeechMOS"
    text = readme_text("/tmp/offline_utmos", "/tmp/offline_utmos/hub/tarepan_SpeechMOS_master", "utmos22_strong")
    assert "export TORCH_HOME=/tmp/offline_utmos" in text
    assert "--utmos_model_dir /tmp/offline_utmos/hub/tarepan_SpeechMOS_master" in text
    assert "utmos22_strong" in text
    print("download_utmos self_test ok")


def parse_args():
    parser = argparse.ArgumentParser("Download UTMOS torch.hub assets for offline use")
    parser.add_argument("--self_test", action="store_true", help="Run helper-function self test and exit")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true", help="Reuse a non-empty output directory")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.self_test:
        self_test()
    else:
        download_utmos(parsed_args)
