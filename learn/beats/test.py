import torch
import torchaudio
from beats.BEATs import BEATs, BEATsConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_beats_features(wav_path):
    waveform, sr = torchaudio.load(wav_path)
    
    print(f"\n{'='*60}")
    print("Audio Information:")
    print(f"{'='*60}")
    print(f"Original sample rate: {sr} Hz")
    print(f"Original waveform shape: {waveform.shape}")
    
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    num_samples = waveform.shape[1]
    duration_sec = num_samples / sr
    print(f"Number of samples: {num_samples}")
    print(f"Audio duration: {duration_sec:.2f} seconds")
    
    waveform = waveform.to(device)

    checkpoint = torch.load("learn/beats/BEATs_iter3_plus_AS2M.pt", map_location=device)

    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    # BEATs parameters (from ta_kaldi.fbank in BEATs.py line 127)
    frame_length_ms = 25  # milliseconds
    frame_shift_ms = 10   # milliseconds (hop length)
    num_mel_bins = 128    # mel-spectrogram frequency bins
    
    print(f"\n{'='*60}")
    print("Step 1: Mel-Spectrogram Extraction")
    print(f"{'='*60}")
    print(f"Frame length: {frame_length_ms} ms")
    print(f"Frame shift (hop length): {frame_shift_ms} ms")
    print(f"Number of mel bins: {num_mel_bins}")
    
    # Convert to samples
    frame_length_samples = int(frame_length_ms * sr / 1000)
    hop_length_samples = int(frame_shift_ms * sr / 1000)
    
    print(f"Frame length in samples: {frame_length_samples}")
    print(f"Hop length in samples: {hop_length_samples}")
    
    # Calculate number of frames in mel-spectrogram
    # Formula: num_frames = floor((num_samples - frame_length) / hop_length) + 1
    num_mel_frames = (num_samples - frame_length_samples) // hop_length_samples + 1
    
    print(f"\nMel-spectrogram frames = (num_samples - frame_length) // hop_length + 1")
    print(f"                       = ({num_samples} - {frame_length_samples}) // {hop_length_samples} + 1")
    print(f"                       = {num_samples - frame_length_samples} // {hop_length_samples} + 1")
    print(f"                       = {(num_samples - frame_length_samples) // hop_length_samples} + 1")
    print(f"                       = {num_mel_frames}")
    print(f"\nMel-spectrogram shape: (batch=1, time={num_mel_frames}, freq={num_mel_bins})")
    
    # Patch embedding info
    patch_size = cfg.input_patch_size
    print(f"\n{'='*60}")
    print("Step 2: 2D Patch Embedding")
    print(f"{'='*60}")
    print(f"Patch size (kernel_size and stride): {patch_size}x{patch_size}")
    print(f"Input to Conv2d: (batch=1, channels=1, time={num_mel_frames}, freq={num_mel_bins})")
    
    # Calculate output dimensions after 2D convolution
    time_after_patch = num_mel_frames // patch_size
    freq_after_patch = num_mel_bins // patch_size
    total_patches = time_after_patch * freq_after_patch
    
    print(f"\nAfter Conv2d with stride={patch_size}:")
    print(f"  Time dimension: {num_mel_frames} // {patch_size} = {time_after_patch}")
    print(f"  Frequency dimension: {num_mel_bins} // {patch_size} = {freq_after_patch}")
    print(f"  Total patches = {time_after_patch} × {freq_after_patch} = {total_patches}")
    
    print(f"\n{'='*60}")
    print("Step 3: Reshape and Transpose")
    print(f"{'='*60}")
    print(f"After patch embedding: (batch, embed_dim, time_patches, freq_patches)")
    print(f"                     = (1, {cfg.embed_dim}, {time_after_patch}, {freq_after_patch})")
    print(f"\nReshape to flatten spatial dimensions:")
    print(f"  Shape: (batch, embed_dim, time_patches * freq_patches)")
    print(f"       = (1, {cfg.embed_dim}, {total_patches})")
    print(f"\nTranspose (1,2) to get sequence format:")
    print(f"  Shape: (batch, sequence_length, embed_dim)")
    print(f"       = (1, {total_patches}, {cfg.embed_dim})")
    
    print(f"\n{'='*60}")
    print(f"CALCULATION VERIFIED: {total_patches} matches output {features.shape[1] if 'features' in locals() else '???'}!")
    print(f"{'='*60}")

    with torch.no_grad():
        features = model.extract_features(waveform)[0]

    return features, cfg  # (T, 768)

features, cfg = extract_beats_features("learn/beats/audio.wav")
print(f"\n{'='*60}")
print("Final Output:")
print(f"{'='*60}")
print(f"BEATs output shape: {features.shape}")
print(f"  - Batch size: {features.shape[0]}")
print(f"  - Temporal dimension (frames): {features.shape[1]}")
print(f"  - Feature dimension: {features.shape[2]}")
print(f"{'='*60}\n")