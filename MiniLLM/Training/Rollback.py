"""Roll a run back to an earlier checkpoint by deleting the newer step-XXXXX folders.

Training auto-resumes from the newest remaining folder, so after a rollback the next run
continues from `--to`. Milestones are never touched.

Usage:
    python -m MiniLLM.Training.Rollback --repo $HF_USERNAME/minillm-125m-ckpt --phase pretrain --to step-03500
    python -m MiniLLM.Training.Rollback ... --yes        # actually delete (default is a dry run)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from MiniLLM.Training.TrainingUtils import STEP_PREFIX, hub_step_folders, step_number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", help="Checkpoint repo on the Hub")
    parser.add_argument("--phase", default="pretrain", choices=["pretrain", "sft", "dpo"])
    parser.add_argument("--to", required=True, help="Folder to keep as the newest, e.g. step-03500")
    parser.add_argument("--local-dir", default=None, help="Also prune a local checkpoint dir, e.g. checkpoints/pretrain")
    parser.add_argument("--yes", action="store_true", help="Delete for real; without it only prints the plan")
    args = parser.parse_args()

    target_step = step_number(args.to)
    hub_victims: list[str] = []
    api = None
    if args.repo:
        from huggingface_hub import HfApi

        api = HfApi()
        folders = hub_step_folders(api, args.repo, args.phase)
        if f"{args.phase}/{args.to}" not in folders:
            raise SystemExit(f"{args.phase}/{args.to} not found on the Hub; available: {folders}")
        hub_victims = [folder for folder in folders if step_number(folder) > target_step]
    local_victims: list[Path] = []
    if args.local_dir:
        local_victims = [path for path in Path(args.local_dir).glob(f"{STEP_PREFIX}*")
                         if path.is_dir() and step_number(path.name) > target_step]

    if not hub_victims and not local_victims:
        print(f"Nothing newer than {args.to}; the next run already resumes from it.")
        return
    print("Will delete:", *[f"hub:{folder}" for folder in hub_victims], *map(str, local_victims), sep="\n  ")
    if not args.yes:
        print("Dry run. Re-run with --yes to delete.")
        return
    for folder in hub_victims:
        api.delete_folder(folder, repo_id=args.repo, commit_message=f"Rollback: remove {folder}")
    for path in local_victims:
        shutil.rmtree(path)
    print(f"Rolled back to {args.to}.")


if __name__ == "__main__":
    main()
