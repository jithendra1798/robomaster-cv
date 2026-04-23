"""
Canonicalize OBB labels so the LONG axis is always the first edge (e01 >= e12).
Rotates corner ordering by 1 position if the first edge is shorter than the second.
Does not change geometry — only reorders the 4 corner points.

Usage:
    # Dry run (prints stats, writes nothing):
    python scripts/normalize_obb_labels.py --src data/mergeRM/labels --dst data/mergeRM/labels_norm --dry_run

    # Normalize to a new directory (safe — does not touch originals):
    python scripts/normalize_obb_labels.py --src data/mergeRM/labels --dst data/mergeRM/labels_norm

    # Verify: re-run the aspect ratio check on the output and confirm near-0% short-first.
"""
import argparse
import numpy as np
from pathlib import Path


def normalize_label_file(src: Path, dst: Path) -> dict:
    """Normalize one label file. Returns counts for reporting."""
    counts = {"total": 0, "flipped": 0, "square": 0, "skipped": 0}
    lines_out = []

    for line in src.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 9:
            lines_out.append(line)
            counts["skipped"] += 1
            continue

        cls = parts[0]
        pts = np.array([[float(parts[i]), float(parts[i + 1])]
                        for i in [1, 3, 5, 7]])

        e01 = np.linalg.norm(pts[1] - pts[0])
        e12 = np.linalg.norm(pts[2] - pts[1])
        counts["total"] += 1

        if e01 < e12 * 0.95:  # first edge clearly shorter — rotate corners by 1
            pts = np.roll(pts, -1, axis=0)
            counts["flipped"] += 1
        elif abs(e01 - e12) / max(e01, e12) < 0.05:
            counts["square"] += 1  # near-square, leave as-is

        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
        lines_out.append(f"{cls} {coords}")

    dst.write_text("\n".join(lines_out) + "\n")
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source labels root (contains train/val/test subdirs)")
    p.add_argument("--dst", required=True, help="Output labels root (mirrored structure)")
    p.add_argument("--dry_run", action="store_true", help="Print stats only, write nothing")
    args = p.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    files = list(src_root.rglob("*.txt"))
    if not files:
        print(f"No .txt files found under {src_root}")
        return

    total = flipped = square = skipped = 0

    for src_file in sorted(files):
        rel = src_file.relative_to(src_root)
        dst_file = dst_root / rel

        if not args.dry_run:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            counts = normalize_label_file(src_file, dst_file)
        else:
            # Still compute stats without writing
            counts = {"total": 0, "flipped": 0, "square": 0, "skipped": 0}
            for line in src_file.read_text().strip().split("\n"):
                parts = line.split()
                if len(parts) != 9:
                    counts["skipped"] += 1
                    continue
                pts = np.array([[float(parts[i]), float(parts[i + 1])]
                                for i in [1, 3, 5, 7]])
                e01 = np.linalg.norm(pts[1] - pts[0])
                e12 = np.linalg.norm(pts[2] - pts[1])
                counts["total"] += 1
                if e01 < e12 * 0.95:
                    counts["flipped"] += 1
                elif abs(e01 - e12) / max(e01, e12) < 0.05:
                    counts["square"] += 1

        total   += counts["total"]
        flipped += counts["flipped"]
        square  += counts["square"]
        skipped += counts["skipped"]

    print(f"Files processed: {len(files)}")
    print(f"Labels total:    {total}")
    print(f"  Flipped:       {flipped} ({100*flipped/total:.1f}%) — short-first → long-first")
    print(f"  Near-square:   {square}  ({100*square/total:.1f}%) — left as-is")
    print(f"  Already OK:    {total-flipped-square} ({100*(total-flipped-square)/total:.1f}%)")
    print(f"  Skipped (non-OBB lines): {skipped}")
    if args.dry_run:
        print("\n[DRY RUN] Nothing written.")
    else:
        print(f"\nNormalized labels written to: {dst_root}")
        print("Next: update configs/data.yaml to point to the new labels dir, then retrain.")


if __name__ == "__main__":
    main()
