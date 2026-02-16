#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch


def convert(ckpt_path: Path, out_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")["splats"]

    means = ckpt["means"].to(torch.float32).contiguous()
    quats = ckpt["quats"].to(torch.float32).contiguous()
    scales = ckpt["scales"].to(torch.float32).contiguous()
    opacities = ckpt["opacities"].to(torch.float32).contiguous()
    sh0 = ckpt["sh0"].to(torch.float32).contiguous()
    shN = ckpt["shN"].to(torch.float32).contiguous()

    n = means.shape[0]
    s0 = sh0.shape[1]
    sn = shN.shape[1]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as f:
        # Header:
        # - magic: 8 bytes (b'GSRAWV1\\0')
        # - N: uint64
        # - S0: uint64
        # - SN: uint64
        f.write(struct.pack("<8sQQQ", b"GSRAWV1\0", n, s0, sn))
        f.write(means.numpy().tobytes(order="C"))
        f.write(quats.numpy().tobytes(order="C"))
        f.write(scales.numpy().tobytes(order="C"))
        f.write(opacities.numpy().tobytes(order="C"))
        f.write(sh0.numpy().tobytes(order="C"))
        f.write(shN.numpy().tobytes(order="C"))

    print(f"Wrote: {out_path}")
    print(f"means.shape = {tuple(means.shape)}")
    print(f"quats.shape = {tuple(quats.shape)}")
    print(f"scales.shape = {tuple(scales.shape)}")
    print(f"opacities.shape = {tuple(opacities.shape)}")
    print(f"sh0.shape = {tuple(sh0.shape)}")
    print(f"shN.shape = {tuple(shN.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert gsplat .pt checkpoint to simple .gsbin file for C++")
    parser.add_argument("ckpt", type=Path, help="Path to input checkpoint .pt")
    args = parser.parse_args()

    convert(args.ckpt, Path("data/model.gsbin"))


if __name__ == "__main__":
    main()
