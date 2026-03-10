"""
LIBERO 데이터셋 시각화 스크립트
HDF5 파일의 demo를 비디오로 저장합니다.

Usage:
    python visualize_dataset.py
    python visualize_dataset.py --hdf5 path/to/file.hdf5 --demo demo_0 --fps 20
"""

import argparse
import os
import h5py
import numpy as np
import imageio
import glob


def list_demos(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys())
    return demos


def make_video(hdf5_path, demo_key, output_path, fps=20, side_by_side=True):
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][demo_key]
        agentview   = demo["obs"]["agentview_rgb"][:]    # (T, H, W, 3) uint8
        eye_in_hand = demo["obs"]["eye_in_hand_rgb"][:]  # (T, H, W, 3) uint8
        actions     = demo["actions"][:]                 # (T, 7)
        T = agentview.shape[0]

    print(f"  demo: {demo_key}  |  frames: {T}  |  action_dim: {actions.shape[1]}")

    writer = imageio.get_writer(output_path, fps=fps, macro_block_size=1)

    for t in range(T):
        ag  = agentview[t]    # (128, 128, 3)
        eih = eye_in_hand[t]  # (128, 128, 3)

        if side_by_side:
            # 두 카메라를 좌우로 붙여서 한 프레임으로
            frame = np.concatenate([ag, eih], axis=1)  # (128, 256, 3)
        else:
            frame = ag

        writer.append_data(frame)

    writer.close()
    print(f"  저장 완료: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=str, default=None,
                        help="HDF5 파일 경로 (미지정 시 libero_spatial 첫 번째 파일 사용)")
    parser.add_argument("--demo", type=str, default="demo_0",
                        help="demo 키 (default: demo_0)")
    parser.add_argument("--all_demos", action="store_true",
                        help="파일 내 모든 demo를 비디오로 저장")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="video/dataset_viz")
    args = parser.parse_args()

    # HDF5 파일 결정
    if args.hdf5 is None:
        pattern = "libero/datasets/libero_spatial/*.hdf5"
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No HDF5 files found: {pattern}")
        args.hdf5 = files[0]

    print(f"파일: {args.hdf5}")
    os.makedirs(args.out_dir, exist_ok=True)

    # task 이름 (파일명에서 추출)
    task = os.path.basename(args.hdf5).replace("_demo.hdf5", "").replace(".hdf5", "")

    demos = list_demos(args.hdf5)
    print(f"데모 수: {len(demos)}  ({demos[0]} ~ {demos[-1]})")

    if args.all_demos:
        for demo_key in demos:
            out_path = os.path.join(args.out_dir, f"{task}__{demo_key}.mp4")
            make_video(args.hdf5, demo_key, out_path, fps=args.fps)
    else:
        if args.demo not in demos:
            print(f"'{args.demo}' 없음. 사용 가능: {demos}")
            return
        out_path = os.path.join(args.out_dir, f"{task}__{args.demo}.mp4")
        make_video(args.hdf5, args.demo, out_path, fps=args.fps)


if __name__ == "__main__":
    main()
