"""Create/update the static chat Space (Transformers.js in the browser) and point it at the model repo.

Static Spaces are free for every account. The model repo must be public and contain the
`onnx/` folder written by `python -m MiniLLM.OnnxExport`, because visitors' browsers download it.

Usage:
    python Scripts/DeploySpace.py --space $HF_USERNAME/minillm-chat --model-repo $HF_USERNAME/minillm-125m-chat
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True, help="Space repo id, e.g. user/minillm-chat")
    parser.add_argument("--model-repo", required=True, help="Public model repo with onnx/ files the page loads")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    model_files = api.list_repo_files(args.model_repo)
    if "onnx/model_quantized.onnx" not in model_files:
        raise SystemExit(f"{args.model_repo} has no onnx/model_quantized.onnx; run MiniLLM.OnnxExport first")
    if api.model_info(args.model_repo).private:
        raise SystemExit(f"{args.model_repo} is private; browsers cannot download it. Make it public first.")

    api.create_repo(args.space, repo_type="space", space_sdk="static", private=args.private, exist_ok=True)
    api.add_space_variable(args.space, "MODEL_REPO", args.model_repo)
    space_dir = Path(__file__).resolve().parent.parent / "ChatSpace"
    api.upload_folder(folder_path=str(space_dir), repo_id=args.space, repo_type="space")
    print(f"Deployed: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
