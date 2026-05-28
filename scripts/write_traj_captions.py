#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    target = Path(args.target_path)
    render_root = target / "render_results"
    prompt = args.prompt.strip() or "uploaded panorama"
    caption = (
        "A cinematic camera trajectory through the uploaded scene, preserving the visual style, "
        "geometry, lighting, and objects from the source panorama. "
        f"Scene prompt: {prompt}"
    )

    count = 0
    for render in sorted(render_root.glob("*/traj*/render.mp4")):
        out = render.with_name("traj_caption.json")
        if out.exists() and not args.overwrite:
            continue
        out.write_text(json.dumps({"prompt": caption}, indent=2), encoding="utf-8")
        count += 1

    print(f"wrote_captions={count}")


if __name__ == "__main__":
    main()
