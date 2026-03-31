def extract_inventory(obs):
    """Extracts the inventory dictionary from the MineRL observation."""
    # MineRL observations typically have an 'inventory' key containing block/item counts
    inventory = obs.get('inventory', {})
    # Filter out empty items to save LLM context window
    active_inventory = {k: v for k, v in inventory.items() if v > 0}
    return active_inventory

import torch
import torch.nn.functional as F
import numpy as np
from mineclip.mineclip.tokenization import tokenize_batch

def check_mineclip_success(mineclip_model, frame_buffer, target_prompt, device):
    """
    Evaluates if the 16-frame buffer matches the target prompt using MineCLIP.
    """
    if len(frame_buffer) < 16:
        return False
        
    # Stack frames into expected shape (Batch, Frames, Channels, Height, Width)
    video_tensor = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_feats = mineclip_model.forward_image_features(video_tensor)  # [B, L, F]
        video_embed = mineclip_model.forward_video_features(image_feats)   # [B, F]

        # --- UPDATED CODE ---
        # Use the standalone tokenize_batch function with max_length=77 for CLIP
        text_tokens = tokenize_batch([target_prompt], max_length=77, language_model="clip").to(device)
        
        # Use encode_text instead of forward_text_features
        text_embed = mineclip_model.encode_text(text_tokens)
        # --------------------

        similarity = F.cosine_similarity(video_embed, text_embed, dim=-1).item()
    #print(similarity)
        
    return similarity