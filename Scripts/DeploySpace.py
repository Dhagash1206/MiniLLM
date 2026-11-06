"""Create/update the Docker chat Space (FastAPI + JS) and point it at the exported model repo.

Usage:
    python Scripts/DeploySpace.py --space $HF_USERNAME/minillm-chat --model-repo $HF_USERNAME/minillm-125m-chat
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True, help="Space repo id, e.g. user/minillm-chat")
    parser.add_argument("--model-repo", required=True, help="Exported HF model repo the app loads")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    api.add_space_variable(args.space, "MODEL_REPO", args.model_repo)
    space_dir = Path(__file__).resolve().parent.parent / "ChatSpace"
    api.upload_folder(folder_path=str(space_dir), repo_id=args.space, repo_type="space")
    print(f"Deployed: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
