"""Small CLI with explicit project-root and configuration paths."""
import argparse
from pathlib import Path

from .io import download_data, verify_data


def main(argv=None):
    parser = argparse.ArgumentParser(description="ETH/UCY coursework: download, verify, or reproduce A1")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root (default: working directory)")
    parser.add_argument("--config", type=Path, default=Path("configs/week01.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="Download pinned sources and verify SHA-256")
    sub.add_parser("verify", help="Verify local source hashes without network access")
    run_parser = sub.add_parser("run", help="Generate all Week 01 figures, audit results, and conclusion")
    run_parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    if args.command == "download":
        receipt = download_data(root)
        print(f"Verified {len(receipt)} source files")
    elif args.command == "verify":
        verify_data(root)
        print("All source checksums match")
    else:
        if args.download:
            download_data(root)
        from .pipeline import run
        run(root, config_path)


if __name__ == "__main__":
    main()
