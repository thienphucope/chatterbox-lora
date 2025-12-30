import os
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from transformers import pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import librosa
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# Import Chatterbox components
from src.chatterbox.tts import ChatterboxTTS, punc_norm
from src.chatterbox.models.s3gen import S3Gen, S3GEN_SR
from src.chatterbox.models.s3tokenizer import S3_SR
from src.chatterbox.models.voice_encoder import VoiceEncoder
from src.chatterbox.models.tokenizers import EnTokenizer
from src.chatterbox.models.t3.modules.cond_enc import T3Cond

# Add matplotlib imports for metrics tracking
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import threading
import time
from collections import deque

# ============================================================================
# CONFIGURATION
# ============================================================================
AUDIO_DATA_DIR = "./audio_data"
BATCH_SIZE = 8  # Increased batch size since we now support it properly
EPOCHS = 10
LEARNING_RATE = 5e-5
WARMUP_STEPS = 500
MAX_AUDIO_LENGTH = 12.0
MIN_AUDIO_LENGTH = 0.5
LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
GRADIENT_ACCUMULATION_STEPS = 4
SAVE_EVERY_N_STEPS = 200
CHECKPOINT_DIR = "checkpoints_lora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
MAX_TEXT_LENGTH = 500
MAX_SPEECH_TOKENS = 1500  # Maximum speech tokens per sample
VALIDATION_SPLIT = 0.1

# Special token IDs
PAD_TOKEN_ID = 0
IGNORE_INDEX = -100


# ============================================================================
# METRICS TRACKER (unchanged)
# ============================================================================
class MetricsTracker:
    def __init__(self, save_path="training_metrics.png", update_interval=2.0):
        self.save_path = save_path
        self.update_interval = update_interval
        self.metrics = {
            'train_loss': deque(maxlen=1000),
            'val_loss': deque(maxlen=100),
            'learning_rate': deque(maxlen=1000),
            'steps': deque(maxlen=1000),
            'epochs': deque(maxlen=1000),
            'batch_loss': deque(maxlen=100),
            'gradient_norm': deque(maxlen=1000),
            'loss_variance': deque(maxlen=100),
            'time_per_step': deque(maxlen=100),
        }
        self.start_time = time.time()
        self.last_update = 0
        self.running = True
        self.lock = threading.Lock()
        
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('Chatterbox TTS LoRA Training Metrics', fontsize=16, fontweight='bold')
        
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        self._create_initial_plot()
    
    def _create_initial_plot(self):
        self.fig.clf()
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        self.ax_loss = self.fig.add_subplot(gs[0, :2])
        self.ax_lr = self.fig.add_subplot(gs[1, 0])
        self.ax_grad = self.fig.add_subplot(gs[1, 1])
        self.ax_batch = self.fig.add_subplot(gs[1, 2])
        self.ax_variance = self.fig.add_subplot(gs[2, 0])
        self.ax_time = self.fig.add_subplot(gs[2, 1])
        self.ax_info = self.fig.add_subplot(gs[0, 2])
        self.ax_epoch = self.fig.add_subplot(gs[2, 2])
        self.ax_info.axis('off')
        
        for ax in [self.ax_loss, self.ax_lr, self.ax_grad, self.ax_batch,
                   self.ax_variance, self.ax_time, self.ax_epoch]:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.fig.savefig(self.save_path, dpi=100, bbox_inches='tight', facecolor='black')
    
    def add_metrics(self, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                if key in self.metrics and value is not None:
                    self.metrics[key].append(value)
            self.last_update = time.time()
    
    def _update_loop(self):
        while self.running:
            time.sleep(self.update_interval)
            if time.time() - self.last_update < self.update_interval * 2:
                self._update_plot()
    
    def _update_plot(self):
        with self.lock:
            try:
                for ax in [self.ax_loss, self.ax_lr, self.ax_grad, self.ax_batch,
                          self.ax_variance, self.ax_time, self.ax_epoch]:
                    ax.clear()
                
                if len(self.metrics['train_loss']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['train_loss']):]
                    self.ax_loss.plot(steps, list(self.metrics['train_loss']),
                                     'b-', label='Train Loss', linewidth=2)
                    self.ax_loss.set_ylim(bottom=0)
                
                if len(self.metrics['val_loss']) > 0:
                    val_steps = list(self.metrics['steps'])[-len(self.metrics['val_loss']):]
                    self.ax_loss.plot(val_steps[-len(self.metrics['val_loss']):],
                                     list(self.metrics['val_loss']),
                                     'r-o', label='Val Loss', linewidth=2, markersize=8)
                
                self.ax_loss.legend()
                self.ax_loss.set_title('Training & Validation Loss', fontweight='bold')
                self.ax_loss.grid(True, alpha=0.3)
                
                if len(self.metrics['learning_rate']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['learning_rate']):]
                    self.ax_lr.plot(steps, list(self.metrics['learning_rate']), 'g-', linewidth=2)
                    self.ax_lr.set_title('Learning Rate', fontweight='bold')
                    self.ax_lr.grid(True, alpha=0.3)
                
                if len(self.metrics['gradient_norm']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['gradient_norm']):]
                    self.ax_grad.plot(steps, list(self.metrics['gradient_norm']), 'm-', linewidth=2)
                    self.ax_grad.set_title('Gradient Norm', fontweight='bold')
                    self.ax_grad.grid(True, alpha=0.3)
                
                if len(self.metrics['batch_loss']) > 0:
                    recent_losses = list(self.metrics['batch_loss'])
                    self.ax_batch.plot(recent_losses, 'c-', linewidth=2)
                    self.ax_batch.axhline(y=np.mean(recent_losses), color='yellow',
                                         linestyle='--', label=f'Mean: {np.mean(recent_losses):.4f}')
                    self.ax_batch.legend()
                    self.ax_batch.set_title('Recent Batch Losses', fontweight='bold')
                    self.ax_batch.grid(True, alpha=0.3)
                
                self.ax_info.clear()
                self.ax_info.axis('off')
                
                info_text = [
                    f"Training Information",
                    f"{'='*25}",
                    f"Device: {DEVICE}",
                    f"Batch Size: {BATCH_SIZE}",
                    f"Grad Accum: {GRADIENT_ACCUMULATION_STEPS}",
                    f"LoRA Rank: {LORA_RANK}",
                ]
                
                if len(self.metrics['steps']) > 0:
                    info_text.append(f"Step: {self.metrics['steps'][-1]}")
                if len(self.metrics['train_loss']) > 0:
                    info_text.append(f"Loss: {self.metrics['train_loss'][-1]:.4f}")
                
                elapsed_time = time.time() - self.start_time
                info_text.append(f"Time: {elapsed_time/3600:.2f}h")
                
                self.ax_info.text(0.05, 0.95, '\n'.join(info_text),
                                 transform=self.ax_info.transAxes,
                                 fontsize=10, verticalalignment='top',
                                 fontfamily='monospace',
                                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
                
                self.fig.savefig(self.save_path, dpi=100, bbox_inches='tight', facecolor='black')
                
            except Exception as e:
                print(f"Error updating plot: {e}")
    
    def stop(self):
        self.running = False
        self.update_thread.join()
        plt.close(self.fig)


# ============================================================================
# LORA IMPLEMENTATION
# ============================================================================
class LoRALayer(nn.Module):
    """LoRA adapter layer"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / np.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)
        
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0/np.sqrt(rank))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.lora_dropout(x)
        result = result @ self.lora_A.T @ self.lora_B.T
        return result * self.scaling


def inject_lora_layers(model: nn.Module, target_modules: List[str], rank: int, alpha: float, dropout: float):
    lora_layers = {}
    device = next(model.parameters()).device
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                if min(module.in_features, module.out_features) < rank:
                    continue
                    
                lora_layer = LoRALayer(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                ).to(device)
                
                lora_layers[name] = lora_layer
                
                original_forward = module.forward
                def make_new_forward(orig_forward, lora):
                    def new_forward(x):
                        return orig_forward(x) + lora(x)
                    return new_forward
                
                module.forward = make_new_forward(original_forward, lora_layer)
    
    return lora_layers


# ============================================================================
# DATA STRUCTURES
# ============================================================================
@dataclass
class AudioSample:
    """Container for audio sample data"""
    audio_path: Path
    transcript: str
    duration: float
    sample_rate: int


@dataclass
class BatchData:
    """Container for properly padded batch data"""
    # Audio tensors
    audio: torch.Tensor              # (B, max_audio_len) - S3GEN_SR
    audio_16k: torch.Tensor          # (B, max_audio_16k_len) - S3_SR (16kHz)
    
    # Lengths for masking
    audio_lengths: torch.Tensor      # (B,) actual audio lengths
    audio_16k_lengths: torch.Tensor  # (B,) actual 16k audio lengths
    
    # Masks
    audio_mask: torch.Tensor         # (B, max_audio_len) - 1 for valid, 0 for pad
    audio_16k_mask: torch.Tensor     # (B, max_audio_16k_len)
    
    # Text
    texts: List[str]
    audio_paths: List[str]


# ============================================================================
# DATASET
# ============================================================================
class TTSDataset(Dataset):
    """Dataset handling with proper length tracking"""
    def __init__(
        self,
        samples: List[AudioSample],
        tokenizer: EnTokenizer,
        s3_sr: int = S3_SR,
        s3gen_sr: int = S3GEN_SR,
        max_audio_length: float = MAX_AUDIO_LENGTH,
        max_text_length: int = MAX_TEXT_LENGTH,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.s3_sr = s3_sr
        self.s3gen_sr = s3gen_sr
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio at S3GEN sample rate
        audio, sr = librosa.load(sample.audio_path, sr=self.s3gen_sr)
        audio = librosa.util.normalize(audio)
        
        # Store original length before any padding
        max_samples = int(self.max_audio_length * self.s3gen_sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        actual_audio_len = len(audio)
        
        # Resample to 16kHz for S3 tokenizer
        audio_16k = librosa.resample(audio, orig_sr=self.s3gen_sr, target_sr=self.s3_sr)
        actual_audio_16k_len = len(audio_16k)
        
        # Process text
        text = punc_norm(sample.transcript)
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
        
        return {
            'audio': torch.FloatTensor(audio),
            'audio_16k': torch.FloatTensor(audio_16k),
            'audio_len': actual_audio_len,
            'audio_16k_len': actual_audio_16k_len,
            'text': text,
            'audio_path': str(sample.audio_path),
        }


def collate_fn(samples: List[Dict]) -> BatchData:
    """
    Custom collate function that properly pads sequences and creates masks.
    """
    batch_size = len(samples)
    
    # Get max lengths in this batch
    max_audio_len = max(s['audio_len'] for s in samples)
    max_audio_16k_len = max(s['audio_16k_len'] for s in samples)
    
    # Initialize tensors
    audio_batch = torch.zeros(batch_size, max_audio_len)
    audio_16k_batch = torch.zeros(batch_size, max_audio_16k_len)
    audio_mask = torch.zeros(batch_size, max_audio_len)
    audio_16k_mask = torch.zeros(batch_size, max_audio_16k_len)
    audio_lengths = torch.zeros(batch_size, dtype=torch.long)
    audio_16k_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    texts = []
    audio_paths = []
    
    for i, sample in enumerate(samples):
        # Audio at S3GEN_SR
        audio_len = sample['audio_len']
        audio_batch[i, :audio_len] = sample['audio'][:audio_len]
        audio_mask[i, :audio_len] = 1.0
        audio_lengths[i] = audio_len
        
        # Audio at 16kHz
        audio_16k_len = sample['audio_16k_len']
        audio_16k_batch[i, :audio_16k_len] = sample['audio_16k'][:audio_16k_len]
        audio_16k_mask[i, :audio_16k_len] = 1.0
        audio_16k_lengths[i] = audio_16k_len
        
        texts.append(sample['text'])
        audio_paths.append(sample['audio_path'])
    
    return BatchData(
        audio=audio_batch,
        audio_16k=audio_16k_batch,
        audio_lengths=audio_lengths,
        audio_16k_lengths=audio_16k_lengths,
        audio_mask=audio_mask,
        audio_16k_mask=audio_16k_mask,
        texts=texts,
        audio_paths=audio_paths,
    )


# ============================================================================
# BATCH CONDITIONAL PREPARATION
# ============================================================================
def prepare_batch_conditionals(
    batch: BatchData,
    model: ChatterboxTTS,
    ve: VoiceEncoder,
    s3gen: S3Gen,
) -> Tuple[T3Cond, List[dict], torch.Tensor]:
    """
    Prepare conditioning for the batch.
    Returns T3Cond, S3Gen references, and valid sample mask.
    """
    B = batch.audio.size(0)
    device = model.device
    
    # Track which samples are valid
    valid_samples = torch.ones(B, dtype=torch.bool, device=device)
    
    # Voice encoder embeddings
    ve_embeds = []
    for i in range(B):
        try:
            # Get actual audio length
            audio_len = batch.audio_16k_lengths[i].item()
            wav_16k = batch.audio_16k[i, :audio_len].numpy()
            
            # Ensure minimum length (0.5 seconds)
            min_samples = S3_SR // 2
            if len(wav_16k) < min_samples:
                wav_16k = np.pad(wav_16k, (0, min_samples - len(wav_16k)), mode='reflect')
            
            utt_embeds = ve.embeds_from_wavs(
                [wav_16k],
                sample_rate=S3_SR,
                as_spk=False,
                batch_size=8,
                rate=1.3,
                overlap=0.5
            )
            
            parts = torch.from_numpy(utt_embeds)
            ref = parts[0].unsqueeze(0)
            sims = F.cosine_similarity(parts, ref, dim=-1)
            voiced = parts[sims > 0.6]
            ve_embed = voiced.mean(0, keepdim=True) if len(voiced) else parts.mean(0, keepdim=True)
            ve_embeds.append(ve_embed)
            
        except Exception as e:
            print(f"Error in voice embedding {i}: {e}")
            valid_samples[i] = False
            # Create zero embedding as placeholder
            ve_embed = torch.zeros(1, 256)
            ve_embeds.append(ve_embed)
    
    ve_embeds = torch.cat(ve_embeds, dim=0).to(device)
    
    # S3Gen reference embeddings
    s3gen_refs = []
    for i in range(B):
        try:
            audio_len = batch.audio_lengths[i].item()
            audio = batch.audio[i, :audio_len].numpy()
            
            # Get reference audio (first DEC_COND_LEN samples)
            ref_len = min(model.DEC_COND_LEN, len(audio))
            ref_audio = audio[:ref_len]
            
            # Pad if needed
            if len(ref_audio) < model.DEC_COND_LEN:
                ref_audio = np.pad(ref_audio, (0, model.DEC_COND_LEN - len(ref_audio)), mode='constant')
            
            s3gen_refs.append(s3gen.embed_ref(ref_audio, S3GEN_SR, device=device))
            
        except Exception as e:
            print(f"Error in S3Gen ref {i}: {e}")
            valid_samples[i] = False
            ref_audio = np.zeros(model.DEC_COND_LEN)
            s3gen_refs.append(s3gen.embed_ref(ref_audio, S3GEN_SR, device=device))
    
    # Speech conditioning tokens
    t3_tokzr = s3gen.tokenizer
    plen = model.t3.hp.speech_cond_prompt_len
    tok_list = []
    tok_masks = []
    
    if plen:
        for i in range(B):
            try:
                audio_16k_len = batch.audio_16k_lengths[i].item()
                wav_16k = batch.audio_16k[i, :audio_16k_len].numpy()
                
                # Get conditioning audio (first ENC_COND_LEN samples)
                ref_len = min(model.ENC_COND_LEN, len(wav_16k))
                ref_16k = wav_16k[:ref_len]
                
                # Ensure minimum length
                min_len = S3_SR // 2
                if len(ref_16k) < min_len:
                    ref_16k = np.pad(ref_16k, (0, min_len - len(ref_16k)), mode='reflect')
                
                tokens, _ = t3_tokzr.forward([ref_16k], max_len=plen)
                
                if not isinstance(tokens, torch.Tensor):
                    tokens = torch.from_numpy(tokens)
                
                tokens = torch.atleast_2d(tokens)
                
                # Create mask for actual tokens
                actual_len = tokens.size(-1)
                mask = torch.ones(1, plen)
                
                # Pad if needed
                if actual_len < plen:
                    tokens = F.pad(tokens, (0, plen - actual_len), value=PAD_TOKEN_ID)
                    mask[:, actual_len:] = 0
                elif actual_len > plen:
                    tokens = tokens[:, :plen]
                
                tok_list.append(tokens)
                tok_masks.append(mask)
                
            except Exception as e:
                print(f"Error tokenizing speech {i}: {e}")
                valid_samples[i] = False
                tok_list.append(torch.zeros(1, plen, dtype=torch.long))
                tok_masks.append(torch.zeros(1, plen))
        
        t3_cond_tokens = torch.cat(tok_list, dim=0).to(device)
        t3_cond_mask = torch.cat(tok_masks, dim=0).to(device)
    else:
        t3_cond_tokens = torch.empty(B, 0, dtype=torch.long, device=device)
        t3_cond_mask = torch.empty(B, 0, device=device)
    
    t3_cond = T3Cond(
        speaker_emb=ve_embeds,
        cond_prompt_speech_tokens=t3_cond_tokens,
        emotion_adv=0.5 * torch.ones(B, 1, 1, device=device),
    )
    
    return t3_cond, s3gen_refs, valid_samples


# ============================================================================
# LOSS COMPUTATION WITH PROPER MASKING
# ============================================================================
def compute_loss(
    model: ChatterboxTTS,
    batch: BatchData,
    t3_cond: T3Cond,
    s3gen_refs: List[dict],
    valid_samples: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute loss with proper masking for variable-length sequences.
    
    Returns:
        loss: The computed loss tensor
        metrics: Dictionary of additional metrics
    """
    batch_size = batch.audio.size(0)
    device = model.device
    
    # =========================================================================
    # 1. TOKENIZE TEXT
    # =========================================================================
    text_tokens_list = []
    text_lengths = []
    
    for i in range(batch_size):
        text = batch.texts[i]
        tokens = model.tokenizer.text_to_tokens(text).to(device)
        text_tokens_list.append(tokens.squeeze(0) if tokens.dim() > 1 else tokens)
        text_lengths.append(tokens.numel())
    
    # Get max text length and pad
    max_text_len = max(text_lengths)
    text_pad_id = getattr(model.tokenizer, 'pad_token_id', PAD_TOKEN_ID)
    
    text_tokens_padded = torch.full(
        (batch_size, max_text_len), 
        text_pad_id, 
        dtype=torch.long, 
        device=device
    )
    text_attention_mask = torch.zeros(batch_size, max_text_len, device=device)
    
    for i, (tokens, length) in enumerate(zip(text_tokens_list, text_lengths)):
        text_tokens_padded[i, :length] = tokens[:length]
        text_attention_mask[i, :length] = 1.0
    
    # Add start/stop tokens
    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    
    # Prepend start token
    text_tokens_padded = F.pad(text_tokens_padded, (1, 0), value=sot)
    text_attention_mask = F.pad(text_attention_mask, (1, 0), value=1.0)
    
    # Append end token (at actual end of each sequence)
    for i, length in enumerate(text_lengths):
        if length + 1 < text_tokens_padded.size(1):
            text_tokens_padded[i, length + 1] = eot
            text_attention_mask[i, length + 1] = 1.0
    
    # =========================================================================
    # 2. TOKENIZE SPEECH
    # =========================================================================
    s3_tokzr = model.s3gen.tokenizer
    MAX_TOKENIZER_SEC = 30
    MAX_TOKENIZER_SAMPLES = MAX_TOKENIZER_SEC * S3_SR
    
    speech_tokens_list = []
    speech_lengths = []
    
    for i in range(batch_size):
        try:
            audio_16k_len = batch.audio_16k_lengths[i].item()
            audio_16k = batch.audio_16k[i, :audio_16k_len].cpu().numpy()
            
            # Truncate if too long
            if len(audio_16k) > MAX_TOKENIZER_SAMPLES:
                audio_16k = audio_16k[:MAX_TOKENIZER_SAMPLES]
            
            # Ensure minimum length
            min_len = S3_SR // 2
            if len(audio_16k) < min_len:
                audio_16k = np.pad(audio_16k, (0, min_len - len(audio_16k)), mode='constant')
            
            tokens, _ = s3_tokzr.forward([audio_16k])
            
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.from_numpy(tokens)
            
            tokens = tokens.squeeze()
            if tokens.dim() == 0:
                tokens = tokens.unsqueeze(0)
            
            # Limit to max speech tokens
            if tokens.numel() > MAX_SPEECH_TOKENS:
                tokens = tokens[:MAX_SPEECH_TOKENS]
            
            speech_tokens_list.append(tokens)
            speech_lengths.append(tokens.numel())
            
        except Exception as e:
            print(f"Error tokenizing speech {i}: {e}")
            # Create dummy tokens
            speech_tokens_list.append(torch.zeros(10, dtype=torch.long))
            speech_lengths.append(10)
    
    # Pad speech tokens
    max_speech_len = max(speech_lengths)
    
    # Use IGNORE_INDEX for padding to automatically mask in loss
    speech_tokens_padded = torch.full(
        (batch_size, max_speech_len),
        IGNORE_INDEX,
        dtype=torch.long,
        device=device
    )
    speech_attention_mask = torch.zeros(batch_size, max_speech_len, device=device)
    
    for i, (tokens, length) in enumerate(zip(speech_tokens_list, speech_lengths)):
        speech_tokens_padded[i, :length] = tokens[:length].to(device)
        speech_attention_mask[i, :length] = 1.0
    
    # =========================================================================
    # 3. PREPARE FOR TRANSFORMER
    # =========================================================================
    # For teacher forcing, input is all tokens except last, target is all except first
    input_speech_tokens = speech_tokens_padded[:, :-1].clone()
    target_speech_tokens = speech_tokens_padded[:, 1:].clone()
    
    # Replace IGNORE_INDEX with 0 for input embeddings (will be masked anyway)
    input_speech_tokens[input_speech_tokens == IGNORE_INDEX] = 0
    
    # Create combined attention mask for input
    # Shape: (batch, seq_len) where seq_len = cond_len + text_len + speech_len - 1
    input_speech_mask = speech_attention_mask[:, :-1]
    
    # =========================================================================
    # 4. FORWARD PASS
    # =========================================================================
    # Prepare input embeddings
    try:
        embeds, len_cond = model.t3.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens_padded,
            speech_tokens=input_speech_tokens,
        )
    except Exception as e:
        print(f"Error in prepare_input_embeds: {e}")
        return torch.tensor(0.0, requires_grad=True, device=device), {'error': 1.0}
    
    # Create full attention mask
    # Conditioning tokens are always attended to
    cond_mask = torch.ones(batch_size, len_cond, device=device)
    
    # Combined mask: [cond_mask, text_mask, speech_mask]
    full_attention_mask = torch.cat([
        cond_mask,
        text_attention_mask,
        input_speech_mask
    ], dim=1)
    
    # Create causal attention mask with padding
    seq_len = embeds.size(1)
    
    # Causal mask: (1, 1, seq_len, seq_len)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device) * float('-inf'),
        diagonal=1
    )
    
    # Combine with padding mask: (batch, 1, 1, seq_len)
    # Positions with mask=0 should have -inf attention
    padding_mask = (1.0 - full_attention_mask).unsqueeze(1).unsqueeze(2) * float('-inf')
    
    # Forward through transformer
    if DEVICE == 'cuda':
        with torch.cuda.amp.autocast():
            outputs = model.t3.tfmr(
                inputs_embeds=embeds,
                attention_mask=full_attention_mask,  # Let model handle masking
            )
            hidden_states = outputs[0]
    else:
        outputs = model.t3.tfmr(
            inputs_embeds=embeds,
            attention_mask=full_attention_mask,
        )
        hidden_states = outputs[0]
    
    # =========================================================================
    # 5. EXTRACT SPEECH LOGITS AND COMPUTE LOSS
    # =========================================================================
    # Speech predictions start after conditioning and text
    speech_start_idx = len_cond + text_tokens_padded.size(1)
    speech_end_idx = speech_start_idx + input_speech_tokens.size(1)
    
    # Ensure we don't go out of bounds
    if speech_end_idx > hidden_states.size(1):
        speech_end_idx = hidden_states.size(1)
        print(f"Warning: Adjusted speech_end_idx to {speech_end_idx}")
    
    if speech_start_idx >= speech_end_idx:
        print(f"Error: Invalid speech indices [{speech_start_idx}:{speech_end_idx}]")
        return torch.tensor(0.0, requires_grad=True, device=device), {'error': 1.0}
    
    # Extract speech hidden states
    speech_hidden = hidden_states[:, speech_start_idx:speech_end_idx]
    
    # Get logits from speech head
    speech_logits = model.t3.speech_head(speech_hidden)
    
    # Ensure target length matches
    actual_pred_len = speech_logits.size(1)
    target_len = target_speech_tokens.size(1)
    
    if actual_pred_len != target_len:
        min_len = min(actual_pred_len, target_len)
        speech_logits = speech_logits[:, :min_len]
        target_speech_tokens = target_speech_tokens[:, :min_len]
    
    # =========================================================================
    # 6. COMPUTE MASKED LOSS
    # =========================================================================
    # Flatten for cross entropy
    logits_flat = speech_logits.reshape(-1, speech_logits.size(-1))
    targets_flat = target_speech_tokens.reshape(-1)
    
    # Cross entropy with ignore_index handles padding automatically
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=IGNORE_INDEX,
        reduction='mean'
    )
    
    # Calculate metrics
    with torch.no_grad():
        # Count valid tokens
        valid_tokens = (targets_flat != IGNORE_INDEX).sum().item()
        total_tokens = targets_flat.numel()
        
        # Calculate accuracy on valid tokens
        predictions = logits_flat.argmax(dim=-1)
        valid_mask = targets_flat != IGNORE_INDEX
        correct = ((predictions == targets_flat) & valid_mask).sum().item()
        accuracy = correct / max(valid_tokens, 1)
        
        # Calculate perplexity
        perplexity = torch.exp(loss).item() if not torch.isnan(loss) else float('inf')
    
    metrics = {
        'valid_tokens': valid_tokens,
        'total_tokens': total_tokens,
        'accuracy': accuracy,
        'perplexity': perplexity,
        'valid_ratio': valid_tokens / max(total_tokens, 1),
    }
    
    # Sanity checks
    if torch.isnan(loss):
        print("ERROR: NaN loss detected!")
        print(f"  Valid tokens: {valid_tokens}, Total: {total_tokens}")
        return torch.tensor(0.0, requires_grad=True, device=device), {'error': 1.0}
    
    if torch.isinf(loss):
        print("ERROR: Infinite loss detected!")
        return torch.tensor(0.0, requires_grad=True, device=device), {'error': 1.0}
    
    return loss, metrics


# ============================================================================
# AUDIO LOADING
# ============================================================================
def load_audio_samples(audio_dir: str, whisper_model) -> List[AudioSample]:
    """Load audio files and generate transcripts using Whisper"""
    samples = []
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    cache_file = Path(audio_dir) / "transcripts_cache.json"
    transcript_cache = {}
    
    if cache_file.exists():
        print(f"Loading transcript cache from {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            transcript_cache = json.load(f)
    
    print(f"Loading audio files from {audio_dir}...")
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(Path(audio_dir).rglob(f"*{ext}"))
    
    print(f"Found {len(audio_files)} audio files")
    
    cache_updated = False
    
    for audio_path in tqdm(audio_files, desc="Processing audio"):
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            duration = len(audio) / sr
            
            if duration < MIN_AUDIO_LENGTH or duration > MAX_AUDIO_LENGTH:
                continue
            
            try:
                relative_path = audio_path.relative_to(Path(audio_dir))
                audio_path_str = relative_path.as_posix()
            except ValueError:
                audio_path_str = str(audio_path.name)
            
            if audio_path_str in transcript_cache:
                transcript = transcript_cache[audio_path_str]['transcript']
            else:
                print(f"\nTranscribing {audio_path.name}...")
                result = whisper_model(str(audio_path), return_timestamps=True)
                transcript = result['text'].strip()
                
                transcript_cache[audio_path_str] = {
                    'transcript': transcript,
                    'duration': duration,
                    'sample_rate': sr
                }
                cache_updated = True
            
            if transcript:
                samples.append(AudioSample(
                    audio_path=audio_path,
                    transcript=transcript,
                    duration=duration,
                    sample_rate=sr
                ))
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    if cache_updated:
        print(f"Saving transcript cache to {cache_file}")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(transcript_cache, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully loaded {len(samples)} samples")
    return samples


# ============================================================================
# CHECKPOINT FUNCTIONS
# ============================================================================
def save_checkpoint(
    model: ChatterboxTTS,
    lora_layers: Dict[str, LoRALayer],
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
    loss: float,
    checkpoint_dir: str,
    is_best: bool = False,
):
    """Save training checkpoint"""
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch{epoch}_step{step}.pt"
    if is_best:
        checkpoint_path = Path(checkpoint_dir) / "best_model.pt"
    
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    lora_state_dict = {}
    for name, layer in lora_layers.items():
        lora_state_dict[f"{name}.lora_A"] = layer.lora_A.data.cpu()
        lora_state_dict[f"{name}.lora_B"] = layer.lora_B.data.cpu()
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'lora_state_dict': lora_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': {
            'lora_rank': LORA_RANK,
            'lora_alpha': LORA_ALPHA,
            'lora_dropout': LORA_DROPOUT,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: ChatterboxTTS,
    lora_layers: Dict[str, LoRALayer],
    optimizer: torch.optim.Optimizer = None,
    scheduler = None,
    device: str = 'cuda',
):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    for name, layer in lora_layers.items():
        if f"{name}.lora_A" in checkpoint['lora_state_dict']:
            layer.lora_A.data = checkpoint['lora_state_dict'][f"{name}.lora_A"].to(device)
            layer.lora_B.data = checkpoint['lora_state_dict'][f"{name}.lora_B"].to(device)
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']


def merge_lora_weights(model: ChatterboxTTS, lora_layers: Dict[str, LoRALayer]):
    """Merge LoRA weights into the base model"""
    with torch.no_grad():
        for name, lora_layer in lora_layers.items():
            parts = name.split('.')
            module = model.t3.tfmr
            for part in parts[:-1]:
                module = getattr(module, part)
            linear_layer = getattr(module, parts[-1])
            
            lora_update = (lora_layer.lora_B @ lora_layer.lora_A) * lora_layer.scaling
            linear_layer.weight.data += lora_update
    
    return model


def save_lora_adapter(lora_layers: Dict[str, LoRALayer], filepath: str):
    """Save LoRA adapter weights and configuration"""
    adapter_dict = {
        'lora_config': {
            'rank': LORA_RANK,
            'alpha': LORA_ALPHA,
            'dropout': LORA_DROPOUT,
            'target_modules': list(set(name.split('.')[-1] for name in lora_layers.keys())),
        },
        'lora_weights': {},
    }
    
    for name, layer in lora_layers.items():
        adapter_dict['lora_weights'][name] = {
            'lora_A': layer.lora_A.cpu(),
            'lora_B': layer.lora_B.cpu(),
        }
    
    torch.save(adapter_dict, filepath)
    print(f"Saved LoRA adapter to {filepath}")


def load_lora_adapter(model: ChatterboxTTS, filepath: str, device: str = 'cuda'):
    """Load LoRA adapter weights"""
    adapter_dict = torch.load(filepath, map_location=device)
    config = adapter_dict['lora_config']
    
    lora_layers = inject_lora_layers(
        model.t3.tfmr,
        config['target_modules'],
        rank=config['rank'],
        alpha=config['alpha'],
        dropout=config['dropout']
    )
    
    for name, weights in adapter_dict['lora_weights'].items():
        if name in lora_layers:
            lora_layers[name].lora_A.data = weights['lora_A'].to(device)
            lora_layers[name].lora_B.data = weights['lora_B'].to(device)
    
    return lora_layers


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def main():
    """Main training function"""
    print(f"="*60)
    print(f"Starting Chatterbox TTS LoRA Fine-tuning")
    print(f"="*60)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"LoRA Rank: {LORA_RANK}, Alpha: {LORA_ALPHA}")
    print(f"="*60)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(save_path="training_metrics.png", update_interval=2.0)
    
    # Load Whisper model
    print("\nLoading Whisper model...")
    whisper_model = pipeline("automatic-speech-recognition", model=WHISPER_MODEL, device="cuda")
    
    # Load audio samples
    samples = load_audio_samples(AUDIO_DATA_DIR, whisper_model)
    if len(samples) == 0:
        raise ValueError(f"No valid audio samples found in {AUDIO_DATA_DIR}")
    
    # Free Whisper memory
    whisper_model.model.cpu()
    del whisper_model
    torch.cuda.empty_cache()
    
    # Split into train/val
    random.shuffle(samples)
    val_size = max(1, int(len(samples) * VALIDATION_SPLIT))
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]
    
    print(f"\nTrain samples: {len(train_samples)}, Validation samples: {len(val_samples)}")
    
    # Load Chatterbox model
    print("\nLoading Chatterbox TTS model...")
    # model = ChatterboxTTS.from_pretrained(DEVICE)
    model = ChatterboxTTS.from_local("./viterbox", DEVICE)
    
    # Inject LoRA layers
    print("\nInjecting LoRA layers...")
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_layers = inject_lora_layers(
        model.t3.tfmr,
        target_modules,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT
    )
    print(f"Injected {len(lora_layers)} LoRA layers")
    
    # Create datasets
    train_dataset = TTSDataset(train_samples, model.tokenizer)
    val_dataset = TTSDataset(val_samples, model.tokenizer)
    
    # Set num_workers
    num_workers = 0 if os.name == 'nt' else 4
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if DEVICE == 'cuda' else False,
        drop_last=True,  # Drop incomplete batches for consistent batch size
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if DEVICE == 'cuda' else False,
    )
    
    # Setup optimizer
    lora_params = []
    for layer in lora_layers.values():
        lora_params.extend([layer.lora_A, layer.lora_B])
    
    optimizer = AdamW(lora_params, lr=LEARNING_RATE, weight_decay=0.01)
    
    total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=LEARNING_RATE * 0.01
    )
    
    # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"Total steps: {total_steps}")
    
    global_step = 0
    best_val_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler() if DEVICE == 'cuda' else None
    
    for epoch in range(EPOCHS):
        # Training
        model.t3.train()
        train_loss = 0.0
        train_steps = 0
        recent_losses = []
        step_start_time = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Prepare conditionals
                t3_cond, s3gen_refs, valid_samples = prepare_batch_conditionals(
                    batch, model, model.ve, model.s3gen
                )
                
                # Compute loss
                loss, metrics = compute_loss(model, batch, t3_cond, s3gen_refs, valid_samples)
                
                if 'error' in metrics:
                    print(f"Skipping batch {batch_idx} due to error")
                    continue
                
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                
                # Track batch loss
                batch_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
                recent_losses.append(batch_loss)
                if len(recent_losses) > 100:
                    recent_losses.pop(0)
                
                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    # Gradient clipping
                    if scaler:
                        scaler.unscale_(optimizer)
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                    
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    scheduler.step()
                    
                    global_step += 1
                    train_loss += batch_loss
                    train_steps += 1
                    
                    # Calculate metrics
                    step_time = time.time() - step_start_time
                    step_start_time = time.time()
                    
                    avg_loss = train_loss / train_steps
                    current_lr = scheduler.get_last_lr()[0]
                    loss_variance = np.var(recent_losses) if len(recent_losses) > 1 else 0
                    
                    # Update metrics tracker
                    metrics_tracker.add_metrics(
                        train_loss=avg_loss,
                        learning_rate=current_lr,
                        steps=global_step,
                        epochs=epoch,
                        batch_loss=batch_loss,
                        gradient_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        loss_variance=loss_variance,
                        time_per_step=step_time
                    )
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'acc': f'{metrics.get("accuracy", 0):.2%}',
                    })
                    
                    # Save checkpoint
                    if global_step % SAVE_EVERY_N_STEPS == 0:
                        save_checkpoint(
                            model, lora_layers, optimizer, scheduler,
                            epoch, global_step, avg_loss, CHECKPOINT_DIR
                        )
                        
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Validation
        model.t3.eval()
        val_loss = 0.0
        val_steps = 0
        val_accuracy = 0.0
        
        print(f"\nRunning validation...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                try:
                    t3_cond, s3gen_refs, valid_samples = prepare_batch_conditionals(
                        batch, model, model.ve, model.s3gen
                    )
                    loss, metrics = compute_loss(model, batch, t3_cond, s3gen_refs, valid_samples)
                    
                    if 'error' not in metrics:
                        val_loss += loss.item()
                        val_accuracy += metrics.get('accuracy', 0)
                        val_steps += 1
                except Exception as e:
                    print(f"Validation error: {e}")
                    continue
        
        avg_val_loss = val_loss / max(val_steps, 1)
        avg_val_accuracy = val_accuracy / max(val_steps, 1)
        avg_train_loss = train_loss / max(train_steps, 1)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {avg_val_accuracy:.2%}")
        
        # Update validation metrics
        metrics_tracker.add_metrics(val_loss=avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                model, lora_layers, optimizer, scheduler,
                epoch, global_step, avg_val_loss, CHECKPOINT_DIR,
                is_best=True
            )
            print(f"  New best model saved! (Val Loss: {best_val_loss:.4f})")
        
        # Save epoch checkpoint
        save_checkpoint(
            model, lora_layers, optimizer, scheduler,
            epoch, global_step, avg_val_loss, CHECKPOINT_DIR
        )
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    # Stop metrics tracker
    metrics_tracker.stop()
    
    # Save final LoRA adapter
    final_adapter_path = Path(CHECKPOINT_DIR) / "final_lora_adapter.pt"
    save_lora_adapter(lora_layers, str(final_adapter_path))
    
    print(f"\nFinal LoRA adapter saved to: {final_adapter_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"\nTo load the LoRA adapter:")
    print(f"  lora_layers = load_lora_adapter(model, '{final_adapter_path}')")


if __name__ == "__main__":
    main()
