#!/usr/bin/env python3
"""
Checkpoint Comparison Utility

Compare two PyTorch checkpoints for equivalence. Handles both compiled
(_orig_mod. prefix) and non-compiled checkpoint formats.

Usage:
    python compare_checkpoints.py checkpoint1.pt checkpoint2.pt [--atol 1e-6] [--rtol 1e-5]

Functions for integration:
    - compare_checkpoints(path1, path2): Full comparison with detailed report
    - assert_checkpoints_equal(path1, path2): Raises AssertionError if not equal
    - verify_checkpoint(checkpoint_path, reference_path): For profile_pretrain.py integration
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
from collections import OrderedDict

import torch


def normalize_key(key: str) -> str:
    """Remove _orig_mod. prefix from compiled model keys."""
    if key.startswith("_orig_mod."):
        return key[len("_orig_mod."):]
    return key


def normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize a state dict by removing _orig_mod. prefixes."""
    return OrderedDict((normalize_key(k), v) for k, v in state_dict.items())


def load_checkpoint(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Load a checkpoint and normalize its keys."""
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    return normalize_state_dict(state_dict)


def compare_tensors(
    t1: torch.Tensor,
    t2: torch.Tensor,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> Tuple[bool, Optional[str]]:
    """
    Compare two tensors for equivalence.

    Returns:
        (is_equal, error_message) - error_message is None if equal
    """
    if t1.shape != t2.shape:
        return False, f"Shape mismatch: {t1.shape} vs {t2.shape}"

    if t1.dtype != t2.dtype:
        return False, f"Dtype mismatch: {t1.dtype} vs {t2.dtype}"

    # Convert to float32 for comparison (bfloat16 doesn't support all ops)
    t1_f = t1.float()
    t2_f = t2.float()

    if not torch.allclose(t1_f, t2_f, atol=atol, rtol=rtol):
        diff = (t1_f - t2_f).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        rel_diff = (diff / (t2_f.abs() + 1e-10)).max().item()
        return False, f"Values differ: max_abs_diff={max_diff:.2e}, mean_abs_diff={mean_diff:.2e}, max_rel_diff={rel_diff:.2e}"

    return True, None


def compare_checkpoints(
    path1: Union[str, Path],
    path2: Union[str, Path],
    atol: float = 1e-6,
    rtol: float = 1e-5,
    verbose: bool = True,
) -> Tuple[bool, Dict]:
    """
    Compare two checkpoints for equivalence.

    Args:
        path1: Path to first checkpoint
        path2: Path to second checkpoint
        atol: Absolute tolerance for tensor comparison
        rtol: Relative tolerance for tensor comparison
        verbose: Print detailed comparison results

    Returns:
        (is_equal, report) where report contains detailed comparison info
    """
    path1, path2 = Path(path1), Path(path2)

    report = {
        "path1": str(path1),
        "path2": str(path2),
        "equal": False,
        "num_params": 0,
        "matching_params": 0,
        "mismatched_params": [],
        "missing_in_1": [],
        "missing_in_2": [],
        "errors": [],
    }

    # Load checkpoints
    try:
        ckpt1 = load_checkpoint(path1)
    except Exception as e:
        report["errors"].append(f"Failed to load {path1}: {e}")
        return False, report

    try:
        ckpt2 = load_checkpoint(path2)
    except Exception as e:
        report["errors"].append(f"Failed to load {path2}: {e}")
        return False, report

    keys1 = set(ckpt1.keys())
    keys2 = set(ckpt2.keys())

    report["missing_in_1"] = sorted(keys2 - keys1)
    report["missing_in_2"] = sorted(keys1 - keys2)

    common_keys = keys1 & keys2
    report["num_params"] = len(common_keys)

    if verbose:
        print(f"Comparing checkpoints:")
        print(f"  {path1}")
        print(f"  {path2}")
        print(f"  Common parameters: {len(common_keys)}")
        if report["missing_in_1"]:
            print(f"  Missing in checkpoint 1: {report['missing_in_1']}")
        if report["missing_in_2"]:
            print(f"  Missing in checkpoint 2: {report['missing_in_2']}")
        print()

    # Compare each parameter
    all_match = True
    for key in sorted(common_keys):
        t1, t2 = ckpt1[key], ckpt2[key]
        is_equal, error = compare_tensors(t1, t2, atol=atol, rtol=rtol)

        if is_equal:
            report["matching_params"] += 1
            if verbose:
                print(f"  [OK] {key}")
        else:
            all_match = False
            report["mismatched_params"].append({"key": key, "error": error})
            if verbose:
                print(f"  [MISMATCH] {key}: {error}")

    # Check for missing keys
    if report["missing_in_1"] or report["missing_in_2"]:
        all_match = False

    report["equal"] = all_match

    if verbose:
        print()
        if all_match:
            print(f"RESULT: Checkpoints are EQUIVALENT ({report['matching_params']}/{report['num_params']} parameters match)")
        else:
            print(f"RESULT: Checkpoints are NOT equivalent")
            print(f"  Matching: {report['matching_params']}/{report['num_params']}")
            print(f"  Mismatched: {len(report['mismatched_params'])}")
            print(f"  Missing in 1: {len(report['missing_in_1'])}")
            print(f"  Missing in 2: {len(report['missing_in_2'])}")

    return all_match, report


def assert_checkpoints_equal(
    path1: Union[str, Path],
    path2: Union[str, Path],
    atol: float = 1e-6,
    rtol: float = 1e-5,
    message: str = "",
) -> None:
    """
    Assert that two checkpoints are equivalent. Raises AssertionError if not.

    Args:
        path1: Path to first checkpoint
        path2: Path to second checkpoint
        atol: Absolute tolerance for tensor comparison
        rtol: Relative tolerance for tensor comparison
        message: Additional message for assertion error
    """
    is_equal, report = compare_checkpoints(path1, path2, atol=atol, rtol=rtol, verbose=False)

    if not is_equal:
        error_parts = [message] if message else []
        error_parts.append(f"Checkpoints are not equivalent:")
        error_parts.append(f"  Path 1: {report['path1']}")
        error_parts.append(f"  Path 2: {report['path2']}")

        if report["mismatched_params"]:
            error_parts.append(f"  Mismatched parameters ({len(report['mismatched_params'])}):")
            for item in report["mismatched_params"][:5]:  # Show first 5
                error_parts.append(f"    - {item['key']}: {item['error']}")
            if len(report["mismatched_params"]) > 5:
                error_parts.append(f"    ... and {len(report['mismatched_params']) - 5} more")

        if report["missing_in_1"]:
            error_parts.append(f"  Missing in checkpoint 1: {report['missing_in_1'][:3]}...")
        if report["missing_in_2"]:
            error_parts.append(f"  Missing in checkpoint 2: {report['missing_in_2'][:3]}...")

        raise AssertionError("\n".join(error_parts))


def verify_checkpoint(
    checkpoint_path: Union[str, Path],
    reference_path: Union[str, Path],
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> bool:
    """
    Verify a checkpoint against a reference. Designed for use in profile_pretrain.py.

    Args:
        checkpoint_path: Path to checkpoint to verify
        reference_path: Path to reference checkpoint
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        True if checkpoints are equivalent, False otherwise

    Prints detailed comparison results to stdout.
    """
    print(f"\n{'='*50}")
    print("CHECKPOINT VERIFICATION")
    print(f"{'='*50}")

    is_equal, report = compare_checkpoints(
        checkpoint_path,
        reference_path,
        atol=atol,
        rtol=rtol,
        verbose=True,
    )

    print(f"{'='*50}\n")
    return is_equal


def compare_all_checkpoints_in_folder(
    folder: Union[str, Path],
    reference_suffix: str = "baseline_eager",
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> Dict[str, bool]:
    """
    Compare all checkpoints in a folder against a reference.

    Args:
        folder: Path to checkpoints folder
        reference_suffix: Suffix to identify reference checkpoint directory
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        Dict mapping checkpoint names to comparison results
    """
    folder = Path(folder)
    results = {}

    # Find reference checkpoint
    reference_dirs = list(folder.glob(f"*{reference_suffix}"))
    if not reference_dirs:
        raise FileNotFoundError(f"No reference checkpoint found with suffix '{reference_suffix}' in {folder}")

    reference_dir = reference_dirs[0]
    reference_ckpt = list(reference_dir.glob("step_*.pt"))[0]

    print(f"Reference checkpoint: {reference_ckpt}")
    print(f"{'='*60}\n")

    # Compare all other checkpoints
    for ckpt_dir in sorted(folder.iterdir()):
        if not ckpt_dir.is_dir() or ckpt_dir == reference_dir:
            continue

        ckpt_files = list(ckpt_dir.glob("step_*.pt"))
        if not ckpt_files:
            continue

        ckpt_path = ckpt_files[0]
        name = ckpt_dir.name

        print(f"\nComparing: {name}")
        print("-" * 40)

        is_equal, _ = compare_checkpoints(
            reference_ckpt,
            ckpt_path,
            atol=atol,
            rtol=rtol,
            verbose=True,
        )
        results[name] = is_equal

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, is_equal in results.items():
        status = "PASS" if is_equal else "FAIL"
        print(f"  [{status}] {name}")

    all_pass = all(results.values())
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare two PyTorch checkpoints for equivalence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare two specific checkpoints
    python compare_checkpoints.py ckpt1.pt ckpt2.pt

    # Compare with custom tolerances
    python compare_checkpoints.py ckpt1.pt ckpt2.pt --atol 1e-5 --rtol 1e-4

    # Compare all checkpoints in folder against baseline
    python compare_checkpoints.py --folder checkpoints/ --reference baseline_eager
        """,
    )

    parser.add_argument("checkpoint1", nargs="?", help="Path to first checkpoint")
    parser.add_argument("checkpoint2", nargs="?", help="Path to second checkpoint")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance (default: 1e-6)")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance (default: 1e-5)")
    parser.add_argument("--folder", type=str, help="Compare all checkpoints in folder")
    parser.add_argument("--reference", type=str, default="baseline_eager", help="Reference checkpoint suffix for folder comparison")

    args = parser.parse_args()

    if args.folder:
        results = compare_all_checkpoints_in_folder(
            args.folder,
            reference_suffix=args.reference,
            atol=args.atol,
            rtol=args.rtol,
        )
        sys.exit(0 if all(results.values()) else 1)

    if not args.checkpoint1 or not args.checkpoint2:
        parser.error("Please provide two checkpoint paths, or use --folder for batch comparison")

    is_equal, _ = compare_checkpoints(
        args.checkpoint1,
        args.checkpoint2,
        atol=args.atol,
        rtol=args.rtol,
        verbose=True,
    )

    sys.exit(0 if is_equal else 1)


if __name__ == "__main__":
    main()
