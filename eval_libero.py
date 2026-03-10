import os
import torch
import numpy as np
import sys
import imageio
from datetime import datetime

# Ensure MambaVLA is in the path
sys.path.append(os.getcwd())

from train_policy import MambaVLATrainingModel
from model_factory import create_mambavla_model
from policy.flowmatching import ActionFLowMatching
from utils.scaler import MinMaxScaler

# LIBERO 관련 임포트 (설치 필요)
try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
except ImportError:
    print("Warning: LIBERO or robosuite not found. Simulation will not run.")

# CKP = "outputs/libero_object/epoch_02000.pt"
CKP = "/home/jeonchanwook/Documents/libero_spatial/final_model.pth"
# TASK = "libero_object"
TASK = "libero_spatial"

def fix_state_dict(state_dict, prefix_to_remove=None):
    new_state_dict = {}
    for k, v in state_dict.items():
        if prefix_to_remove and k.startswith(prefix_to_remove):
            new_key = k[len(prefix_to_remove):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = CKP

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading model from {checkpoint_path}...")

    # 1. 모델 생성
    # Note: Use the same hyperparameters as training
    model = create_mambavla_model(
        camera_names=["agentview", "robot0_eye_in_hand"],
        action_dim=7,
        action_seq_len=10,
        device=device
    )

    # create policy wrapper
    policy = ActionFLowMatching(backbones=model.model.model, device=device)
    eval_model = MambaVLATrainingModel(model=model, policy=policy)

    # 2. 가중치 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        # 기존 libero_object 형식일 경우
        # Based on inspection:
        # checkpoint['model_state_dict'] contains backbone keys (MambaVLAPolicy)
        # checkpoint['policy_state_dict'] contains ActionFLowMatching keys with 'model.module.' prefix
        # AttributeError: 'MambaVLA' object has no attribute 'obs_encoder' 에러 해결을 위한 별칭 추가
        eval_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # Load backbone weights
        print("Loading backbone weights...")
        eval_model.model.model.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # Load policy weights (ActionFLowMatching)
        print("Loading policy weights...")
        policy_sd = fix_state_dict(checkpoint['policy_state_dict'], prefix_to_remove="model.module.")
        # Add 'model.' prefix back because ActionFLowMatching.state_dict() keys start with 'model.'
        fixed_policy_sd = {f"model.{k}": v for k, v in policy_sd.items()}
        eval_model.policy.load_state_dict(fixed_policy_sd, strict=False)
    else:
        # 현재 libero_spatial 형식일 경우 (전체 모델 state_dict)
        eval_model.model.load_state_dict(checkpoint, strict=False)

    eval_model.model.obs_encoder = eval_model.model.img_encoder
    eval_model.model.obs_encoder.camera_names = eval_model.model.cam_names
    eval_model.to(device).float()

    
    # Setup a default scaler if none is available (MinMax with 0-1 range for 7 dims)
    # In real scenarios, you should load the actual scaler used during training.
    dummy_actions = torch.zeros(1, 7)
    eval_model.scaler = MinMaxScaler(dummy_actions, False, device)
    eval_model.model.set_scaler(eval_model.scaler)

    eval_model.to(device)
    eval_model.eval()
    print("Model ready.")

    # 2. LIBERO 벤치마크 환경 설정
    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        libero_object = benchmark_dict[f"{TASK}"]()

        task_id = 0
        task_name = libero_object.get_task_names()[task_id]

        env_args = {
            "bddl_file_name": libero_object.get_task_bddl_file_path(task_id),
            "camera_heights": 128,
            "camera_widths": 128,
        }

        env = OffScreenRenderEnv(**env_args)
        env.seed(42)
        obs = env.reset()

        # 비디오 저장
        video_dir = "video"
        os.makedirs(video_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(video_dir, f"{task_name}_{timestamp}.mp4")
        video_writer = imageio.get_writer(video_path, fps=20)
        print(f"Video will be saved to: {video_path}")

        # 3. 롤아웃 실행
        print(f"Starting evaluation for task: {task_name}")
        max_steps = 600

        with torch.no_grad():
            for step in range(max_steps):
                frame = env.sim.render(width=512, height=512, camera_name="agentview")[::-1, :, :]
                video_writer.append_data(frame)

                # Prepare observation
                obs_dict = {
                    "agentview_image": torch.from_numpy(obs["agentview_image"].copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0,
                    "robot0_eye_in_hand_image": torch.from_numpy(obs["robot0_eye_in_hand_image"].copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0,
                    "lang": task_name
                }

                # Predict action
                action = eval_model.predict(obs_dict)
                action = action.cpu().numpy()

                # Step environment
                obs, reward, done, info = env.step(action)

                if reward > 0.5:
                    print(f"Task Succeeded at step {step}!")
                    break
        
        env.close()
    except Exception as e:
        print(f"Simulation error: {e}")

if __name__ == "__main__":
    evaluate()
