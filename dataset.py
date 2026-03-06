"""
LIBERO Dataset for MambaVLA training.

HDF5 structure:
  data/{demo_N}/obs/agentview_rgb:   (T, 128, 128, 3)  uint8
  data/{demo_N}/obs/eye_in_hand_rgb: (T, 128, 128, 3)  uint8
  data/{demo_N}/actions:             (T, 7)            float64
  data/{demo_N}/robot_states:        (T, ...)

Dataset returns per timestep:
  obs_dict = {
      "agentview_image":    Tensor [1, 3, 128, 128]  float32 [0,1]
      "eye_in_hand_image":  Tensor [1, 3, 128, 128]  float32 [0,1]
      "lang":               str
  }
  action  = Tensor [action_seq_len, 7]  (zero-padded at episode end)
  mask    = Tensor [action_seq_len]     (1=valid, 0=padded)
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class LiberoDataset(Dataset):
    """
    Loads all LIBERO HDF5 demo files from a directory.
    Builds a flat index over (file, demo, timestep) tuples.
    """

    camera_names = ["agentview", "eye_in_hand"]

    def __init__(
        self,
        dataset_dir: str,
        action_seq_len: int = 10,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.action_seq_len = action_seq_len

        # Index: list of (hdf5_path, demo_key, step_idx, task_name)
        self._index = []
        self._build_index()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self):
        import h5py

        hdf5_files = sorted(glob.glob(os.path.join(self.dataset_dir, "*.hdf5")))
        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in {self.dataset_dir}")

        for path in hdf5_files:
            task_name = self._task_name_from_path(path)
            with h5py.File(path, "r") as f:
                for demo_key in sorted(f["data"].keys()):
                    T = f["data"][demo_key]["actions"].shape[0]
                    for t in range(T):
                        self._index.append((path, demo_key, t, task_name))

        print(f"[LiberoDataset] {len(self._index)} timesteps from "
              f"{len(hdf5_files)} tasks in '{self.dataset_dir}'")

    @staticmethod
    def _task_name_from_path(path: str) -> str:
        """'pick_up_the_black_bowl_..._demo.hdf5' → 'pick up the black bowl ...'"""
        basename = os.path.basename(path)
        name = basename.replace("_demo.hdf5", "").replace(".hdf5", "")
        return name.replace("_", " ")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        import h5py

        path, demo_key, t, task_name = self._index[idx]

        with h5py.File(path, "r") as f:
            demo = f["data"][demo_key]
            T_total = demo["actions"].shape[0]

            # --- images: (H, W, 3) uint8 → (1, 3, H, W) float [0,1] ---
            agentview = demo["obs"]["agentview_rgb"][t]       # (H, W, 3)
            eye_in_hand = demo["obs"]["eye_in_hand_rgb"][t]   # (H, W, 3)

            # --- actions: [action_seq_len, 7] with zero-padding ---
            end = min(t + self.action_seq_len, T_total)
            actions = demo["actions"][t:end].astype(np.float32)   # (valid_len, 7)

        valid_len = actions.shape[0]
        pad_len = self.action_seq_len - valid_len

        action = np.zeros((self.action_seq_len, 7), dtype=np.float32)
        action[:valid_len] = actions
        if pad_len > 0:
            action[valid_len:] = actions[-1]  # repeat last action

        mask = np.zeros(self.action_seq_len, dtype=np.float32)
        mask[:valid_len] = 1.0

        # --- convert images ---
        agentview_t   = self._to_tensor(agentview)    # (1, 3, H, W)
        eye_in_hand_t = self._to_tensor(eye_in_hand)  # (1, 3, H, W)

        obs_dict = {
            "agentview_image":   agentview_t,
            "eye_in_hand_image": eye_in_hand_t,
            "lang": task_name,
        }

        return obs_dict, torch.from_numpy(action), torch.from_numpy(mask)

    @staticmethod
    def _to_tensor(img_hwc: np.ndarray) -> torch.Tensor:
        """(H, W, 3) uint8 → (1, 3, H, W) float32 in [0, 1]"""
        img = torch.from_numpy(img_hwc).permute(2, 0, 1).float() / 255.0
        return img.unsqueeze(0)  # add T=1 dim

    # ------------------------------------------------------------------
    # Required by Trainer for scaler initialization
    # ------------------------------------------------------------------

    def get_all_actions(self) -> torch.Tensor:
        """Return all actions across the entire dataset for scaler fitting."""
        import h5py

        seen_files = {}
        all_actions = []

        for path, demo_key, t, _ in self._index:
            if (path, demo_key) in seen_files:
                continue
            seen_files[(path, demo_key)] = True
            with h5py.File(path, "r") as f:
                acts = f["data"][demo_key]["actions"][:]
            all_actions.append(torch.from_numpy(acts.astype(np.float32)))

        return torch.cat(all_actions, dim=0)  # (N_total_steps, 7)
