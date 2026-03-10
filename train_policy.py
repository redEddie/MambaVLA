import os
import sys
import pickle
import random
import logging
import warnings
from typing import Any, Optional, Dict
from utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
# Suppress all warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Filter stderr for robosuite/gym warnings
if not hasattr(sys.stderr, '_is_filtered'):
    _original_stderr = sys.stderr
    
    class FilteredStderr:
        def __init__(self, original):
            self.original = original
            self._is_filtered = True
        
        def write(self, text):
            if any(kw in text for kw in ['[robosuite WARNING]', 'No private macro', 'Gym has been unmaintained', 'upgrade to Gymnasium', 'Scope.user.*setter']):
                return
            self.original.write(text)
        
        def flush(self):
            self.original.flush()
        
        def __getattr__(self, name):
            return getattr(self.original, name)
    
    sys.stderr = FilteredStderr(_original_stderr)

try:
    import wandb
    WANDB_AVAILABLE = True
except (ImportError, Exception):
    WANDB_AVAILABLE = False

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

import multiprocessing as mp

from model_factory import create_mambavla_model
from mambavla_model import MambaVLA
from policy.flowmatching import ActionFLowMatching
from utils.scaler import Scaler, ActionScaler, MinMaxScaler
from utils.ema import ExponentialMovingAverage

# Set multiprocessing start method to 'spawn' to avoid CUDA issues
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

log = logging.getLogger(__name__)


class MambaVLATrainingModel(nn.Module):
    """
    Wrapper model that combines MambaVLA with Flow Matching policy
    for use with the Trainer class.
    """
    
    def __init__(
        self,
        model: MambaVLA,
        policy: ActionFLowMatching,
        use_lr_scheduler: bool = False,
        learning_rate: float = 1e-4,
        sampling_steps: int = 4,
    ):
        super().__init__()
        self.model = model
        self.policy = policy
        self.use_lr_scheduler = use_lr_scheduler
        self.learning_rate = learning_rate
        self.scaler = None
        self.sampling_steps = sampling_steps  # For action generation during inference
        self.working_dir = None  # Will be set by trainer
        
    def set_scaler(self, scaler):
        """Set the data scaler."""
        self.scaler = scaler
    
    def configure_optimizers(self):
        """
        Initialize optimizers and learning rate schedulers based on model configuration.
        """
        # Default optimizer config values
        weight_decay = 0.05
        betas = [0.9, 0.999]
        
        # Optimize all model parameters (encoder, obs_encoder, embeddings, action_pred, language_encoder, etc.)
        # Note: self.model.parameters() already includes language_encoder if it exists, so no need to add separately
        optim_groups = [
            {"params": self.model.parameters(), "weight_decay": weight_decay},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=betas)

        # Optionally initialize the scheduler
        if self.use_lr_scheduler:
            # Note: lr_scheduler config would need to be passed to __init__ if needed
            # For now, return (optimizer, None) to match expected tuple format
            # The scheduler can be added later when lr_scheduler config is available
            return optimizer, None
        else:
            return optimizer
    
    def forward(self, obs_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns loss.
        Matches working version's MambaVLA.forward() behavior.
        
        Args:
            obs_dict: Dictionary with observation data
                     Should contain camera images (keys ending with '_image') and 'lang_emb'
            action: Action tensor [B, T, D]
        
        Returns:
            Loss tensor
        """
        device = next(self.model.parameters()).device
        
        # Process language embedding first (match working version)
        # If "lang" (text) is provided, use language encoder; otherwise use pre-computed "lang_emb"
        if "lang" in obs_dict and self.model.language_encoder is not None:
            # Process raw language text through CLIP encoder
            lang_text = obs_dict["lang"]
            # Handle both list of strings and tensor cases
            if isinstance(lang_text, list):
                lang_emb = self.model.language_encoder(lang_text)  # [B, 1, lang_emb_dim]
            elif isinstance(lang_text, torch.Tensor):
                # If it's already a tensor, might be batched strings - convert to list
                if lang_text.dtype == torch.long:
                    # It's tokenized, skip language encoder
                    lang_emb = obs_dict.get("lang_emb")
                    if lang_emb is None:
                        raise ValueError("obs_dict contains tokenized 'lang' but no 'lang_emb'. Use raw text strings instead.")
                else:
                    # Try to process as strings
                    lang_list = [str(item) for item in lang_text.cpu().numpy().flatten()]
                    lang_emb = self.model.language_encoder(lang_list)
            else:
                lang_emb = self.model.language_encoder([str(lang_text)])
            # Remove sequence dimension if needed: [B, 1, lang_emb_dim] -> [B, lang_emb_dim]
            if len(lang_emb.shape) == 3 and lang_emb.shape[1] == 1:
                lang_emb = lang_emb.squeeze(1)
        elif "lang_emb" in obs_dict:
            # Use pre-computed language embedding
            lang_emb = obs_dict["lang_emb"].to(device)
        else:
            raise ValueError("obs_dict must contain either 'lang' (text) or 'lang_emb' (pre-computed embedding)")
        
        lang_emb = lang_emb.to(device)
        
        # Extract image keys from obs_dict using camera names from the encoder
        # Match working version's _input_embeddings method
        camera_names = self.model.obs_encoder.camera_names
        image_obs_dict = {}
        
        # Get first camera to determine batch and time dimensions (like working version)
        first_camera_key = f"{camera_names[0]}_image"
        # Try to find it with different key format
        for key in obs_dict.keys():
            key_base = key.replace('_image', '').replace('_rgb', '').replace('_img', '')
            if key_base == camera_names[0]:
                first_camera_key = key
                break
        
        if first_camera_key not in obs_dict:
            raise KeyError(f"Camera '{camera_names[0]}' not found in observation dictionary. Available keys: {list(obs_dict.keys())}")
        
        first_img = obs_dict[first_camera_key].to(device)
        
        # Handle time dimension like working version: [B, T, C, H, W] -> reshape for encoder
        if len(first_img.shape) == 5:
            B, T, C, H, W = first_img.shape
            # Reshape all images to [B*T, C, H, W] for encoder (match working version line 101)
            for camera_name in camera_names:
                image_key = f"{camera_name}_image"
                # Try to find the key in obs_dict
                found = False
                for key in obs_dict.keys():
                    key_base = key.replace('_image', '').replace('_rgb', '').replace('_img', '')
                    if key_base == camera_name:
                        img = obs_dict[key].to(device)
                        if len(img.shape) == 5:
                            # Reshape [B, T, C, H, W] -> [B*T, C, H, W] (match working version)
                            image_obs_dict[image_key] = img.view(B * T, C, H, W)
                        else:
                            image_obs_dict[image_key] = img
                        found = True
                        break
                
                if not found:
                    raise KeyError(
                        f"Camera '{camera_name}' not found in observation dictionary. "
                        f"Available keys: {list(obs_dict.keys())}"
                    )
        else:
            # Images are already [B, C, H, W] - use as-is
            for camera_name in camera_names:
                image_key = f"{camera_name}_image"
                found = False
                for key in obs_dict.keys():
                    key_base = key.replace('_image', '').replace('_rgb', '').replace('_img', '')
                    if key_base == camera_name:
                        image_obs_dict[image_key] = obs_dict[key].to(device)
                        found = True
                        break
                
                if not found:
                    raise KeyError(
                        f"Camera '{camera_name}' not found in observation dictionary. "
                        f"Available keys: {list(obs_dict.keys())}"
                    )
        
        # Encode images - encoder returns [B*T, num_cameras, latent_dim] or [B, num_cameras, latent_dim]
        encoded_states = self.model.obs_encoder(image_obs_dict)
        
        # Reshape back if we had time dimension: [B*T, num_cameras, latent_dim] -> [B, T*num_cameras, latent_dim]
        if len(first_img.shape) == 5:
            # encoded_states is [B*T, num_cameras, latent_dim]
            num_cameras = encoded_states.shape[1]
            latent_dim = encoded_states.shape[2]
            # Reshape to [B, T, num_cameras, latent_dim] then flatten to [B, T*num_cameras, latent_dim]
            encoded_states = encoded_states.view(B, T, num_cameras, latent_dim)
            encoded_states = encoded_states.view(B, T * num_cameras, latent_dim)
        # If no time dimension, encoded_states is already [B, num_cameras, latent_dim] which is correct
        
        # Handle language embedding shape - match working version
        # Working version expects [B, lang_emb_dim] or [B, 1, lang_emb_dim]
        if len(lang_emb.shape) == 1:
            lang_emb = lang_emb.unsqueeze(0)  # [lang_emb_dim] -> [1, lang_emb_dim]
        if len(lang_emb.shape) == 2:
            # [B, lang_emb_dim] - keep as is, model will handle it
            pass
        
        # Move actions to device
        action = action.to(device)
        
        # Forward pass through Flow Matching - pass full action sequence as-is
        # The action sequence is already prepared correctly in _prepare_batch_data
        loss, _ = self.policy(
            actions=action,
            state=encoded_states,
            lang_embed=lang_emb
        )
        
        return loss
    
    def store_model_weights(self, working_dir: str, sv_name: str = 'model',
                             optimizer=None, epoch: int = 0):
        """Store model weights."""
        os.makedirs(working_dir, exist_ok=True)
        checkpoint_path = os.path.join(working_dir, f'{sv_name}.pt')
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'epoch': epoch,
        }
        if optimizer is not None:
            ckpt['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(ckpt, checkpoint_path)
        return checkpoint_path
    
    def reset(self):
        """Reset model state (called at start of each episode)."""
        # No internal state to reset for now, but method needed for simulator
        pass
    
    @torch.no_grad()
    def predict(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict actions from observations (for inference/evaluation).
        
        Args:
            obs_dict: Dictionary with observation data
                    Should contain camera images and 'lang_emb'
        
        Returns:
            Action tensor [action_dim] (unscaled)
        """
        self.eval()
        device = next(self.model.parameters()).device
        
        # Extract image keys from obs_dict using camera names from the encoder
        camera_names = self.model.obs_encoder.camera_names
        image_obs_dict = {}
        for camera_name in camera_names:
            # Try to find the corresponding key in obs_dict
            # Handle different key formats: "camera_name_image", "robot0_camera_name_image", etc.
            found = False
            possible_keys = [
                f"{camera_name}_image",
                f"robot0_{camera_name}_image",
                camera_name,
                f"robot0_{camera_name}",
            ]
            
            for possible_key in possible_keys:
                if possible_key in obs_dict:
                    img = obs_dict[possible_key].to(device)
                    # Handle shape: [1, 1, C, H, W] -> [1, C, H, W] or [1, C, H, W] -> [1, C, H, W]
                    if len(img.shape) == 5:
                        img = img.squeeze(1)  # Remove time dimension if present
                    image_obs_dict[f"{camera_name}_image"] = img
                    found = True
                    break
            
            # If not found with exact matches, try partial matching
            if not found:
                for key in obs_dict.keys():
                    # Remove common prefixes and suffixes
                    key_base = key.replace('robot0_', '').replace('_image', '').replace('_rgb', '').replace('_img', '')
                    if key_base == camera_name:
                        img = obs_dict[key].to(device)
                        # Handle shape: [1, 1, C, H, W] -> [1, C, H, W]
                        if len(img.shape) == 5:
                            img = img.squeeze(1)
                        image_obs_dict[f"{camera_name}_image"] = img
                        found = True
                        break
            
            if not found:
                raise KeyError(
                    f"Camera '{camera_name}' not found in observation dictionary. "
                    f"Available keys: {list(obs_dict.keys())}"
                )
        
        # Encode images
        encoded_states = self.model.obs_encoder(image_obs_dict)  # [1, obs_tok_len, latent_dim]
        
        # Get language embedding - use language encoder if available
        if 'lang' in obs_dict and self.model.language_encoder is not None:
            # Process raw language text through CLIP encoder
            lang_text = obs_dict['lang']
            if isinstance(lang_text, list):
                lang_emb = self.model.language_encoder(lang_text)
            elif isinstance(lang_text, str):
                lang_emb = self.model.language_encoder([lang_text])
            else:
                # Try to convert to string
                lang_emb = self.model.language_encoder([str(lang_text)])
            # Remove sequence dimension if needed: [1, 1, lang_emb_dim] -> [1, lang_emb_dim]
            if len(lang_emb.shape) == 3 and lang_emb.shape[1] == 1:
                lang_emb = lang_emb.squeeze(1)
        elif 'lang_emb' in obs_dict:
            lang_emb = obs_dict['lang_emb'].to(device)
        else:
            raise ValueError("obs_dict must contain either 'lang' (text) or 'lang_emb' (pre-computed embedding)")
        
        lang_emb = lang_emb.to(device)
        
        # Handle different shapes
        if len(lang_emb.shape) == 1:
            lang_emb = lang_emb.unsqueeze(0).unsqueeze(0)
        elif len(lang_emb.shape) == 2:
            lang_emb = lang_emb.unsqueeze(1)
        
        # Create noise actions for generation
        action_dim = self.model.action_dim
        action_seq_len = self.model.action_seq_len
        noise_actions = torch.randn(1, action_seq_len, action_dim).to(device)
        
        # Generate actions using Flow Matching
        # Use config default of 4 sampling steps (can be overridden via kwargs)
        sampling_steps = getattr(self, 'sampling_steps', 4)
        generated_actions = self.policy.generate_actions(
            noise_actions=noise_actions,
            state=encoded_states,
            lang_embed=lang_emb,
            sample_steps=sampling_steps
        )  # [1, action_seq_len, action_dim]
        
        # Take the first action (or last if action_seq_len > 1)
        if action_seq_len > 1:
            action = generated_actions[0, -1, :]  # [action_dim]
        else:
            action = generated_actions[0, 0, :]  # [action_dim]
        
        # Unscale action if scaler is available
        if self.scaler is not None:
            action = self.scaler.inverse_scale_output(action.unsqueeze(0)).squeeze(0)
        
        return action
    
    @property
    def get_model_state(self):
        """Get model state for multiprocessing (returns tuple of state_dict and scaler)."""
        model_state = self.model.state_dict()
        # Return scaler object itself (not state_dict) as expected by simulator
        scaler_obj = self.scaler if self.scaler is not None else None
        return (model_state, scaler_obj)


class Trainer:
    """Basic train/test class to be inherited."""

    def __init__(
            self,
            training_dataset: Any,
            validation_dataset: Any = None,
            training_batch_size: int = 512,
            validation_batch_size: int = 512,
            dataloader_workers: int = 8,
            device: str = 'cpu',
            total_epochs: int = 100,
            enable_data_scaling: bool = True,
            data_scaler_type: str = "minmax",
            evaluation_frequency: int = 50,
            observation_sequence_length: int = 1,
            ema_decay_rate: float = 0.999,
            enable_ema: bool = False,
            checkpoint_frequency: int = 10,
            eval_during_training: Optional[int] = None,
            eval_callback: Optional[Any] = None,
            resume_path: Optional[str] = None,
    ):
        """Initialize."""

        # Dataset and data loading configuration
        self.trainset = training_dataset
        self.valset = validation_dataset
        self.train_batch_size = training_batch_size
        self.val_batch_size = validation_batch_size
        self.num_workers = dataloader_workers

        # Training configuration
        self.epoch = total_epochs
        self.perception_seq_len = observation_sequence_length
        self.eval_every_n_epochs = evaluation_frequency
        self.save_every_n_epochs = checkpoint_frequency

        # Device and environment configuration
        self.device = device
        self.working_dir = os.getcwd()

        # Data scaling configuration
        self.scale_data = enable_data_scaling
        self.scaling_type = data_scaler_type

        # EMA configuration
        self.decay_ema = ema_decay_rate
        self.if_use_ema = enable_ema

        # Evaluation during training configuration
        self.eval_during_training = eval_during_training
        self.eval_callback = eval_callback

        # Resume configuration
        self.resume_path = resume_path
        self.start_epoch = 0

        # Initialize data loaders
        self._setup_data_loaders()

        # Initialize scaler
        self._setup_scaler()

        log.info("Number of training samples: {}".format(len(self.trainset)))

    def _setup_data_loaders(self):
        """Setup training and validation data loaders."""
        self.train_dataloader = DataLoader(
            self.trainset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=4 if self.num_workers > 0 else None
        )

        if self.valset is not None:
            self.val_dataloader = DataLoader(
                self.valset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=False
            )

    def _setup_scaler(self):
        """Setup data scaler based on configuration."""
        if hasattr(self.trainset, 'get_all_actions'):
            actions = self.trainset.get_all_actions()
        else:
            # Fallback: try to get actions from dataset
            log.warning("Dataset doesn't have get_all_actions method. Using default scaler.")
            # Create dummy actions for scaler initialization
            actions = torch.zeros(100, 7)  # Default shape
        
        if self.scaling_type == 'minmax':
            self.scaler = MinMaxScaler(actions, self.scale_data, self.device)
        else:
            self.scaler = ActionScaler(actions, self.scale_data, self.device)

    def main(self, model):
        """Run main training/testing pipeline."""
        self._setup_training_components(model)
        self._run_training_loop(model)
        self._finalize_training(model)

    def _setup_training_components(self, model):
        """Setup scaler, EMA, and optimizer for training."""
        model.set_scaler(self.scaler)

        if self.if_use_ema:
            self.ema_helper = ExponentialMovingAverage(model.parameters(), self.decay_ema, self.device)

        if model.use_lr_scheduler:
            result = model.configure_optimizers()
            if isinstance(result, tuple):
                self.optimizer, self.scheduler = result
            else:
                self.optimizer = result
                self.scheduler = None
        else:
            self.optimizer = model.configure_optimizers()
            self.scheduler = None

        # Resume from checkpoint
        if self.resume_path is not None:
            log.info(f"Resuming from checkpoint: {self.resume_path}")
            ckpt = torch.load(self.resume_path, map_location=self.device)
            model.model.load_state_dict(ckpt['model_state_dict'])
            model.policy.load_state_dict(ckpt['policy_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.start_epoch = ckpt.get('epoch', 0)
            log.info(f"Resumed from epoch {self.start_epoch}")

    def _run_training_loop(self, model):
        """Execute the main training loop over all epochs."""
        for num_epoch in tqdm(range(self.start_epoch, self.epoch), desc="Epochs", dynamic_ncols=True):
            epoch_loss = self._train_single_epoch(model, num_epoch)
            self._log_epoch_results(num_epoch, epoch_loss)
            self._save_checkpoint_if_needed(model, num_epoch)
            self._evaluate_if_needed(model, num_epoch)

    def _train_single_epoch(self, model, num_epoch):
        """Train for a single epoch and return the average loss."""
        epoch_loss = torch.tensor(0.0).to(self.device)
        num_batches = 0
        for data in tqdm(self.train_dataloader, desc="Batches", leave=False, dynamic_ncols=True):
            obs_dict, action, mask = data
            
            obs_dict, action = self._prepare_batch_data(obs_dict, action)
            
            batch_loss = self.train_one_step(model, obs_dict, action)
            
            epoch_loss += batch_loss
            num_batches += 1
            
            # Don't log batch losses separately - they cause step conflicts
            # Only log epoch-level metrics

        avg_loss = epoch_loss / len(self.train_dataloader)
        return avg_loss

    def _prepare_batch_data(self, obs_dict, action):
        """Prepare observation and action data for training."""
        # put data on device - match working version exactly
        for camera in obs_dict.keys():
            if camera == 'lang':
                continue
            
            obs_dict[camera] = obs_dict[camera].to(self.device)

            if 'rgb' not in camera and 'image' not in camera:
                continue
            obs_dict[camera] = obs_dict[camera][:, :self.perception_seq_len].contiguous()

        action = self.scaler.scale_output(action)
        action = action[:, self.perception_seq_len - 1:, :].contiguous()

        return obs_dict, action

    def _log_epoch_results(self, num_epoch, epoch_loss):
        """Log epoch results to wandb and console."""
        log.info("Epoch {}: Mean train loss is {}".format(num_epoch, epoch_loss.item()))
        
        if WANDB_AVAILABLE and wandb.run is not None:
            try:
                # Prepare log dict
                log_dict = {
                    "epoch": num_epoch + 1,  # 1-indexed epoch
                    "train_loss": epoch_loss.item(),
                    "progress": (num_epoch + 1) / self.epoch,
                }
                
                # Add learning rate if scheduler is available
                if self.scheduler is not None:
                    try:
                        current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else None
                        if current_lr is not None:
                            log_dict["learning_rate"] = current_lr
                    except Exception:
                        pass
                
                # Log all metrics together with consistent step
                wandb.log(log_dict, step=num_epoch + 1)  # Use epoch+1 as step (1-indexed)
            except Exception as e:
                log.warning(f"Failed to log to wandb: {e}")

    def _save_checkpoint_if_needed(self, model, num_epoch):
        """Save model checkpoint if it's time to do so."""
        if (num_epoch + 1) % self.save_every_n_epochs == 0:
            try:
                if hasattr(model, 'store_model_weights'):
                    checkpoint_path = model.store_model_weights(
                        self.working_dir, sv_name=f"epoch_{num_epoch + 1:05d}",
                        optimizer=self.optimizer, epoch=num_epoch + 1)
                else:
                    # Fallback: save using torch.save
                    checkpoint_path = os.path.join(self.working_dir, f"epoch_{num_epoch + 1:05d}.pt")
                    torch.save({
                        'epoch': num_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, checkpoint_path)
                log.info(f"Checkpoint saved to {checkpoint_path}")
                
                # Log checkpoint save to wandb
                if WANDB_AVAILABLE and wandb.run is not None:
                    try:
                        wandb.log({"checkpoint/saved": True, "checkpoint/epoch": num_epoch + 1})
                    except Exception:
                        pass
            except Exception as e:
                log.warning(f"Failed to save checkpoint at epoch {num_epoch + 1}: {e}")

    def _evaluate_if_needed(self, model, num_epoch):
        """Run evaluation if it's time to do so."""
        if self.eval_during_training is not None and self.eval_callback is not None:
            if (num_epoch + 1) % self.eval_during_training == 0:
                log.info(f"Running evaluation at epoch {num_epoch + 1}...")
                try:
                    self.eval_callback(model, num_epoch + 1)
                except Exception as e:
                    log.warning(f"Evaluation failed at epoch {num_epoch + 1}: {e}")

    def _finalize_training(self, model):
        """Finalize training by applying EMA and saving final model."""
        log.info("training done")
        
        if self.if_use_ema:
            self.ema_helper.store(model.parameters())
            self.ema_helper.copy_to(model.parameters())
        
        if hasattr(model, 'store_model_weights'):
            model.store_model_weights(model.working_dir, sv_name='final_model',
                                      optimizer=self.optimizer, epoch=self.epoch)
        else:
            final_path = os.path.join(self.working_dir, 'final_model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.epoch,
            }, final_path)


    def train_one_step(self, model, obs_dict, action):
        """Run a single training step."""
        model.train()
        
        loss = model(obs_dict, action)
        
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        
        if model.use_lr_scheduler and self.scheduler is not None:
            self.scheduler.step()

        if self.if_use_ema:
            self.ema_helper.update(model.parameters())
        
        return loss

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters, split='val'):
        """Run a given number of evaluation steps."""
        return None


def train_policy(
    dataloader: Any,
    batch_size: int = 256,
    num_epochs: int = 500,
    learning_rate: float = 1e-4,
    device: Optional[str] = None,
    latent_dim: int = 256,
    embed_dim: int = 256,
    n_layer: int = 5,
    d_intermediate: int = 256,
    obs_tok_len: Optional[int] = None,  # Auto-inferred from dataloader
    action_seq_len: int = 10,
    action_dim: int = 7,
    lang_emb_dim: int = 512,
    save_dir: str = './checkpoints',
    save_freq: int = 10,
    enable_ema: bool = True,
    enable_data_scaling: bool = True,
    data_scaler_type: str = "minmax",
    dataloader_workers: int = 4,
    eval_during_training: Optional[int] = None,
    eval_callback: Optional[Any] = None,
    wandb_project: Optional[str] = "MambaVLA",
    wandb_entity: Optional[str] = None,
    wandb_name: Optional[str] = None,
    model_type: str = "mamba",
    transformer_cfg: Optional[dict] = None,
    resume: Optional[str] = None,
    **kwargs
):
    """
    Train MambaVLA policy with ActionFLowMatching.
    
    Args:
        dataloader: Dataset or DataLoader instance
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to run on (default: auto-detect)
        latent_dim: Dimension of encoded image features
        embed_dim: Embedding dimension for Mamba backbone
        n_layer: Number of Mamba layers
        d_intermediate: Intermediate dimension for MLP
        obs_tok_len: Number of observation tokens (auto-inferred from dataloader if None)
        action_seq_len: Length of action sequence
        action_dim: Dimension of action space
        lang_emb_dim: Dimension of language embeddings
        save_dir: Directory to save checkpoints
        save_freq: Frequency of saving checkpoints
        enable_ema: Enable Exponential Moving Average
        enable_data_scaling: Enable data scaling
        data_scaler_type: Type of scaler ('minmax' or 'action')
        dataloader_workers: Number of dataloader workers
        eval_during_training: Evaluate model every N epochs (None to disable)
        eval_callback: Callback function(model, epoch) to run evaluation
        **kwargs: Additional model arguments
    
    Returns:
        training_model: The training model instance
        trainer: The trainer instance
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"Using device: {device}")
    
    # Initialize wandb if available
    if WANDB_AVAILABLE and wandb_project is not None:
        try:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_name,
                config={
                    # Training
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "device": device,
                    "enable_ema": enable_ema,
                    "enable_data_scaling": enable_data_scaling,
                    "data_scaler_type": data_scaler_type,
                    "save_freq": save_freq,
                    "eval_during_training": eval_during_training,
                    "dataloader_workers": dataloader_workers,
                    "resume": resume,
                    # Model architecture
                    "model_type": model_type,
                    "latent_dim": latent_dim,
                    "embed_dim": embed_dim,
                    "n_layer": n_layer,
                    "d_intermediate": d_intermediate,
                    "action_seq_len": action_seq_len,
                    "action_dim": action_dim,
                    "lang_emb_dim": lang_emb_dim,
                    "sampling_steps": kwargs.get("sampling_steps", 4),
                    "obs_tok_len": obs_tok_len,
                    "transformer_cfg": transformer_cfg,
                    # Optimizer
                    "betas": kwargs.get("betas", [0.9, 0.9]),
                    "transformer_weight_decay": kwargs.get("transformer_weight_decay", 0.05),
                    "obs_encoder_weight_decay": kwargs.get("obs_encoder_weight_decay", 0.05),
                    "ema_decay_rate": kwargs.get("ema_decay_rate", 0.995),
                    # Dataset
                    "dataset_size": len(dataset) if hasattr(dataset, '__len__') else "unknown",
                    "num_demos": dataset.num_data if hasattr(dataset, 'num_data') else "unknown",
                    "max_len_data": dataset.max_len_data if hasattr(dataset, 'max_len_data') else "unknown",
                    "demos_per_task": dataset.demos_per_task if hasattr(dataset, 'demos_per_task') else "unknown",
                }
            )
            log.info("Wandb initialized successfully")
        except Exception as e:
            log.warning(f"Failed to initialize wandb: {e}")
    
    # Get dataset from dataloader if needed
    if hasattr(dataloader, 'dataset'):
        dataset = dataloader.dataset
    else:
        dataset = dataloader
    
    log.info(f"Dataset loaded: {len(dataset)} samples")
    

    log.info(f"Model type: {model_type}")
    model = create_mambavla_model(
        dataloader=dataset,
        camera_names=None,  # Auto-detect from dataloader
        latent_dim=latent_dim,
        action_dim=action_dim,
        lang_emb_dim=lang_emb_dim,
        embed_dim=embed_dim,
        obs_tok_len=obs_tok_len,  # Auto-inferred from camera count if None
        action_seq_len=action_seq_len,
        perception_seq_len=1,
        state_dim=kwargs.get('state_dim', 7),
        device=device,
        n_layer=n_layer,
        d_intermediate=d_intermediate,
        sampling_steps=kwargs.get('sampling_steps', 4),
        transformer_weight_decay=kwargs.get('transformer_weight_decay', 0.05),
        obs_encoder_weight_decay=kwargs.get('obs_encoder_weight_decay', 0.05),
        learning_rate=learning_rate,
        betas=kwargs.get('betas', None),
        model_type=model_type,
        transformer_cfg=transformer_cfg,
    )
    
    # Count total and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    log.info(f"Model created with {total_params:,} total parameters")
    log.info(f"  - Trainable: {trainable_params:,} parameters ({100*trainable_params/total_params:.1f}%)")
    log.info(f"  - Frozen: {frozen_params:,} parameters ({100*frozen_params/total_params:.1f}%)")
    
    # Log model info to wandb
    if WANDB_AVAILABLE and wandb.run is not None:
        try:
            wandb.log({
                "model/num_parameters": total_params,
                "model/trainable_parameters": trainable_params,
                "model/frozen_parameters": frozen_params,
                "model/trainable_percentage": 100*trainable_params/total_params,
                "model/latent_dim": latent_dim,
                "model/embed_dim": embed_dim,
                "model/n_layer": n_layer,
                "model/d_intermediate": d_intermediate,
                "model/action_seq_len": action_seq_len,
            })
        except Exception as e:
            log.warning(f"Failed to log model info to wandb: {e}")
    
    # Create Trainer
    trainer = Trainer(
        training_dataset=dataset,
        validation_dataset=None,
        training_batch_size=batch_size,
        validation_batch_size=batch_size,
        dataloader_workers=dataloader_workers,
        device=device,
        total_epochs=num_epochs,
        enable_data_scaling=enable_data_scaling,
        data_scaler_type=data_scaler_type,
        evaluation_frequency=50,
        observation_sequence_length=1,
        ema_decay_rate=0.995,  # Match config default
        enable_ema=enable_ema,
        checkpoint_frequency=save_freq,
        eval_during_training=eval_during_training,
        eval_callback=eval_callback,
        resume_path=resume,
    )
    
    # Set working directory
    trainer.working_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    model.working_dir = save_dir
    
    # Start training
    log.info("Starting training...")
    trainer.main(model)
    
    log.info("Training completed!")
    return model, trainer
