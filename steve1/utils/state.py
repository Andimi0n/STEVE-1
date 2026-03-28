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

def check_mineclip_success(mineclip_model, frame_buffer, target_prompt, device, threshold=0.25):
    """
    Evaluates if the 16-frame buffer matches the target prompt using MineCLIP.
    """
    if len(frame_buffer) < 16:
        return False
        
    # Stack frames into expected shape (Batch, Frames, Channels, Height, Width)
    # Note: Adjust dimension ordering based on specific MineCLIP implementation expectations
    video_tensor = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get video embedding
        video_embed = mineclip_model.forward_video_features(video_tensor)
        
        # Get text embedding
        text_tokens = mineclip_model.tokenize([target_prompt]).to(device)
        text_embed = mineclip_model.forward_text_features(text_tokens)
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(video_embed, text_embed, dim=-1).item()
        
    return similarity >= threshold