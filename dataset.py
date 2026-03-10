"""
LIBERO Dataset for MambaVLA training.
Follows OpenARM-VLA / KNU-Mamba dataloader structure.

HDF5 structure:
  data/{demo_N}/obs/agentview_rgb:   (T, 128, 128, 3)  uint8
  data/{demo_N}/obs/eye_in_hand_rgb: (T, 128, 128, 3)  uint8
  data/{demo_N}/obs/joint_states:    (T, 7)
  data/{demo_N}/obs/gripper_states:  (T, 2)
  data/{demo_N}/actions:             (T, 7)            float64

Dataset returns per slice:
  obs_dict = {
      "agentview_image":    Tensor [1, 3, 128, 128]  float32 [0,1]
      "eye_in_hand_image":  Tensor [1, 3, 128, 128]  float32 [0,1]
      "lang_emb":           Tensor [1, 512]           float32 (CLIP)
      "robot_states":       Tensor [1, state_dim]     float32
  }
  action  = Tensor [chunck_size, action_dim]
  mask    = Tensor [chunck_size]
"""

import logging
import os
import glob
import pickle

import numpy as np
import torch

log = logging.getLogger(__name__)


def _generate_clip_embeddings(data_dir, output_path):
    """Generate CLIP embeddings for all tasks in the dataset directory."""
    try:
        import clip
    except ImportError:
        raise ImportError("pip install clip (openai-clip) is required to generate embeddings")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)

    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".hdf5")])
    if not files:
        raise FileNotFoundError(f"No .hdf5 files in {data_dir}")

    task_embeddings = {}
    for f in files:
        name = os.path.splitext(f)[0]
        if name.endswith("_demo"):
            name = name[:-5]
        task_name = name.replace("_", " ")
        tokens = clip.tokenize([task_name]).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        task_embeddings[name] = emb.cpu()
        log.info(f"  CLIP embedding: {name} -> {emb.shape}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as fp:
        pickle.dump(task_embeddings, fp)
    log.info(f"Saved embeddings to {output_path}")
    return task_embeddings


class LiberoDataset:
    """
    LIBERO dataset following OpenARM-VLA / KNU-Mamba structure.
    All data is eagerly loaded into memory at init.
    Indexed by (demo_idx, start, start+chunck_size) slices.
    """

    camera_names = ["agentview", "eye_in_hand"]

    def __init__(
        self,
        dataset_dir: str,
        action_seq_len: int = 5,
        max_len_data: int = 136,
        demos_per_task: int = 50,
        start_idx: int = 0,
        action_dim: int = 7,
        state_dim: int = 9,
        embedding_dir: str = None,
    ):
        self.dataset_dir = dataset_dir
        self.action_seq_len = action_seq_len
        self.max_len_data = max_len_data
        self.demos_per_task = demos_per_task
        self.start_idx = start_idx
        self.action_dim = action_dim
        self.state_dim = state_dim

        # --- Load or generate CLIP language embeddings ---
        benchmark_type = os.path.basename(os.path.normpath(dataset_dir))
        if embedding_dir is None:
            embedding_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "language_embeddings"
            )
        emb_path = os.path.join(embedding_dir, f"{benchmark_type}.pkl")

        if not os.path.exists(emb_path):
            log.warning(f"Embeddings not found at {emb_path}, generating...")
            _generate_clip_embeddings(dataset_dir, emb_path)

        with open(emb_path, "rb") as f:
            self.tasks = pickle.load(f)

        # --- Eagerly load all demo data ---
        self.data_embs = []
        self.agentview_rgb = []
        self.eye_in_hand_rgb = []
        self.all_states = []
        actions_list = []
        masks_list = []

        self._load_all_demos(actions_list, masks_list)

        self.actions = torch.from_numpy(np.concatenate(actions_list)).float()
        self.masks = torch.from_numpy(np.concatenate(masks_list)).float()
        self.num_data = len(self.agentview_rgb)

        # --- Build slices ---
        self.slices = self._build_slices()

        log.info(
            f"[LiberoDataset] {self.num_data} demos, "
            f"{len(self.slices)} slices (chunck_size={action_seq_len})"
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_all_demos(self, actions_list, masks_list):
        import h5py

        hdf5_files = sorted(glob.glob(os.path.join(self.dataset_dir, "*.hdf5")))
        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in {self.dataset_dir}")

        for path in hdf5_files:
            filename = os.path.basename(path).split(".")[0]
            if filename.endswith("_demo"):
                filename = filename[:-5]

            # Get or generate task embedding
            if filename not in self.tasks:
                log.warning(f"Task '{filename}' not in embeddings, regenerating...")
                benchmark_type = os.path.basename(os.path.normpath(self.dataset_dir))
                emb_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "language_embeddings"
                )
                emb_path = os.path.join(emb_dir, f"{benchmark_type}.pkl")
                _generate_clip_embeddings(self.dataset_dir, emb_path)
                with open(emb_path, "rb") as fp:
                    self.tasks = pickle.load(fp)

            task_emb = self.tasks[filename]

            with h5py.File(path, "r") as f:
                demo_keys = list(f["data"].keys())
                indices = np.argsort([int(k[5:]) for k in demo_keys])

                end_idx = self.start_idx + self.demos_per_task
                for i in indices[self.start_idx:end_idx]:
                    demo_name = demo_keys[i]
                    demo = f["data"][demo_name]

                    if "num_samples" in demo.attrs:
                        demo_length = int(demo.attrs["num_samples"])
                    else:
                        demo_length = demo["actions"].shape[0]

                    effective_length = min(demo_length, self.max_len_data)
                    if demo_length > self.max_len_data:
                        log.warning(
                            f"Demo {demo_name} length {demo_length} exceeds "
                            f"max_len_data={self.max_len_data}; truncating"
                        )

                    # Actions: zero-padded to max_len_data
                    zero_actions = np.zeros(
                        (1, self.max_len_data, self.action_dim), dtype=np.float32
                    )
                    action_data = demo["actions"][:effective_length].astype(np.float32)
                    zero_actions[0, :effective_length, :] = action_data
                    actions_list.append(zero_actions)

                    # Mask
                    zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)
                    zero_mask[0, :effective_length] = 1
                    masks_list.append(zero_mask)

                    # Images (stored as numpy arrays)
                    self.agentview_rgb.append(
                        demo["obs"]["agentview_rgb"][:effective_length]
                    )
                    self.eye_in_hand_rgb.append(
                        demo["obs"]["eye_in_hand_rgb"][:effective_length]
                    )

                    # Robot states
                    joint_states = demo["obs"]["joint_states"][:effective_length]
                    gripper_states = demo["obs"]["gripper_states"][:effective_length]
                    robot_states = np.concatenate(
                        (joint_states, gripper_states), axis=-1
                    )
                    self.all_states.append(robot_states)

                    # Language embedding
                    self.data_embs.append(task_emb)

            log.info(f"Loaded {path}: {min(len(indices), self.demos_per_task)} demos")

    # ------------------------------------------------------------------
    # Slice building
    # ------------------------------------------------------------------

    def _build_slices(self):
        slices = []
        for i in range(self.num_data):
            T = self.get_seq_length(i)
            if T - self.action_seq_len < 0:
                log.warning(
                    f"Ignored short sequence #{i}: len={T}, "
                    f"window={self.action_seq_len}"
                )
            else:
                slices += [
                    (i, start, start + self.action_seq_len)
                    for start in range(T - self.action_seq_len + 1)
                ]
        return slices

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]

        # Language embedding
        task_emb = self.data_embs[i]
        if isinstance(task_emb, torch.Tensor):
            task_emb = task_emb.detach().cpu().float()
        else:
            task_emb = torch.tensor(task_emb, dtype=torch.float32)

        # Images: single frame at timestep 'start'
        agentview_rgb = torch.from_numpy(
            self.agentview_rgb[i][start : start + 1].copy()
        ).float().permute(0, 3, 1, 2) / 255.0

        eye_in_hand_rgb = torch.from_numpy(
            self.eye_in_hand_rgb[i][start : start + 1].copy()
        ).float().permute(0, 3, 1, 2) / 255.0

        # Robot states
        robot_states = torch.from_numpy(
            self.all_states[i][start : start + 1].copy()
        ).float()

        # Action chunk & mask
        act = self.actions[i, start:end]
        mask = self.masks[i, start:end]

        obs = {
            "agentview_image": agentview_rgb,
            "eye_in_hand_image": eye_in_hand_rgb,
            "lang_emb": task_emb,
            "robot_states": robot_states,
        }

        return obs, act, mask

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_all_actions(self):
        """Return all valid actions concatenated for scaler fitting."""
        result = []
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)
