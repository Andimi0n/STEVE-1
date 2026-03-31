import argparse
import json
import os
import requests
import cv2
import numpy as np
import torch
from collections import deque
from tqdm import tqdm

from steve1.data.text_alignment.vae import load_vae_model
from steve1.utils.mineclip_agent_env_utils import load_mineclip_agent_env
from steve1.utils.text_overlay_utils import created_fitted_text_image
from steve1.utils.video_utils import save_frames_as_video
from steve1.utils.state import extract_inventory, check_mineclip_success

from steve1.config import PRIOR_INFO, DEVICE
from steve1.utils.embed_utils import get_prior_embed

FPS = 20

def create_video_frame(gameplay_pov, prompt):
    """Creates a frame for the generated video with the gameplay POV and the prompt text on the right side."""
    frame = cv2.cvtColor(gameplay_pov, cv2.COLOR_RGB2BGR)
    prompt_section = created_fitted_text_image(frame.shape[1] // 2, prompt,
                                               background_color=(0, 0, 0),
                                               text_color=(255, 255, 255))
    pad_top_height = (frame.shape[0] - prompt_section.shape[0]) // 2
    pad_top = np.zeros((pad_top_height, prompt_section.shape[1], 3), dtype=np.uint8)
    pad_bottom_height = frame.shape[0] - pad_top_height - prompt_section.shape[0]
    pad_bottom = np.zeros((pad_bottom_height, prompt_section.shape[1], 3), dtype=np.uint8)
    prompt_section = np.vstack((pad_top, prompt_section, pad_bottom))
    frame = np.hstack((frame, prompt_section))
    return frame

def get_next_step_mistral(high_level_goal, current_task_status):
    """Queries a local llama model via Ollama to get the next step."""
    system_prompt = """
    You are a Minecraft planning agent. The user wants to achieve a high-level goal.
    Based on the current status, output the NEXT single action STEVE-1 should take.
    The agent is a short term agent so each step should be quite a basic task. For example: "Craft a wooden pickaxe" -> Collect Wood Logs -> Craft Wood Planks -> Craft crafting table -> Craft stick -> Craft wooden pickaxe
    We use a MineCLIP model to evaluate if a "visual" task is completed or we directly check if a specific item is in the inventory for "gather" task.

    Based on the Current Status and Inventory, output valid JSON with exactly these keys:
    1. "steve_prompt": A 2-4 word visual description of the action (e.g., "chopping a tree", "crafting wooden planks", "finding a desert").
    2. "task_type": Must be EXACTLY "gather" (for getting a specific item through collecting, mining, crafting, ...) or "visual" (for exploring/finding).
    3. "target_item": The exact in-game item ID in snake_case (e.g., "log", "planks", "stick", "wooden_pickaxe"). DO NOT use spaces, capital letters, or plurals (except "planks"). Do not specify wood types (use "log", not "oak_log" or "Wood Logs"). Set to null if task_type is "visual".
    4. "target_quantity": The integer amount of the target_item you want to OBTAIN during this specific step (e.g., 1). Set to null if task_type is "visual".
    5. "rationale": Why this is the logical next step.
    """
    
    prompt = f"Goal: {high_level_goal}\nStatus: {current_task_status}"
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.1",
        "prompt": f"{system_prompt}\n\n{prompt}",
        "format": "json",
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response_data = response.json()
        return json.loads(response_data["response"])
    except Exception as e:
        print(f"Error querying llama: {e}")
        return {"steve_prompt": "explore", "task_type": "visual", "target_item": None, "target_quantity": None, "rationale": "Fallback due to error."}

def run_autonomous_mineclip(agent, mineclip, prior, env, high_level_goal, device, output_video_dirpath):
    """Runs the agent using Mistral for planning and MineCLIP for success criteria, saving the video at the end."""
    
    obs = env.reset()
    status = "Just spawned in a new world. No items yet."
    
    frame_buffer = deque(maxlen=16)
    video_frames = []
    frame_prompts = []
    
    window_name = 'STEVE-1 Autonomous Planner'
    
    print("Starting autonomous run. Press 'q' in the video window or Ctrl+C in the terminal to stop and save.")

    try:
        while True:
            # 1. Ask Mistral for the next step
            print(f"\n[LLAMA] Planning next step for goal: {high_level_goal}...")
            print(f"[STATUS] {status}")
            plan = get_next_step_mistral(high_level_goal, status)

            print(plan)
            
            current_prompt = plan.get("steve_prompt", "explore")
            
            task_type = plan.get("task_type", "visual")
            
            target_item = plan.get("target_item")
            target_qty = plan.get("target_quantity", 1) or 1
            
            print(f"[PLANNER] Rationale: {plan.get('rationale', '')}")
            print(f"[PLANNER] Executing Step: '{current_prompt}' | Type: {task_type}")
            if task_type == "gather" and target_item:
                print(f"[PLANNER] Goal: Obtain {target_qty}x {target_item}")
            
            # 2. Embed the prompt using STEVE-1's prior
            prompt_embed = get_prior_embed(current_prompt, mineclip, prior, device)
            
            # 3. Execution loop for the current step
            ticks = 0
            max_ticks = 400 # 20 seconds at 20 FPS
            task_completed = False
            
            # --- HYBRID BASELINES ---
            initial_inventory = extract_inventory(obs)
            baseline_mineclip_score = None
            
            print(f"Running '{current_prompt}'...")
            with torch.cuda.amp.autocast():
                while ticks < max_ticks:
                    # Get action and step env
                    minerl_action = agent.get_action(obs, prompt_embed)
                    
                    obs, reward, done, info = env.step(minerl_action)
                    
                    # Gracefully handle environment termination or errors
                    if done:
                        print("\n[SYSTEM] Environment ended the episode (Agent died or timed out).")
                        break
                    
                    if info and 'error' in info:
                        print(f"\n[SYSTEM] MineRL Error: {info['error']}")
                        break
                    # -----------------------------------------------
                    
                    # Process frame for MineCLIP buffer
                    raw_frame = obs['pov']
                    mineclip_frame = cv2.resize(raw_frame, (256, 160)) 
                    frame_tensor = torch.from_numpy(mineclip_frame).permute(2, 0, 1).float() / 255.0
                    frame_buffer.append(frame_tensor)
                    
                    # Create and store video frame for output
                    disp_frame = create_video_frame(raw_frame, current_prompt)
                    video_frames.append(disp_frame)
                    frame_prompts.append(current_prompt)
                    
                    # Show the frame
                    cv2.imshow(window_name, disp_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nUser requested early exit via 'q'.")
                        raise KeyboardInterrupt
                    
                    # --- HYBRID SUCCESS CHECK ---
                    
                    # GATHERING TASK: Rely on absolute ground truth (Inventory)
                    if task_type == "gather" and target_item:
                        current_inventory = extract_inventory(obs)
                        
                        # Clean up target item just in case Llama hallucinates spaces/plurals
                        target_clean = target_item.lower().replace(" ", "_")
                        # Remove trailing 's' so "logs" becomes "log", but keep "planks" intact
                        if target_clean.endswith('s') and target_clean != "planks":
                            target_clean = target_clean[:-1]
                        
                        # Fuzzy match: sum up ALL inventory items that contain the target_clean string
                        # (This means target "log" will successfully count "oak_log", "birch_log", AND "log")
                        current_qty = sum(count for item, count in current_inventory.items() if target_clean in item)
                        initial_qty = sum(count for item, count in initial_inventory.items() if target_clean in item)
                        
                        # Check if we obtained the required amount during this step
                        if current_qty >= initial_qty + target_qty:
                            print(f"\n[INVENTORY] Success: Gathered/Crafted {target_qty}x '{target_item}' (Matched in inventory via '{target_clean}')!")
                            task_completed = True
                            break
                            
                    # VISUAL TASK: Rely on MineCLIP delta spike
                    elif task_type == "visual" and ticks % 16 == 0 and len(frame_buffer) == 16:
                        current_score = check_mineclip_success(mineclip, frame_buffer, current_prompt, device)
                        
                        if baseline_mineclip_score is None:
                            baseline_mineclip_score = current_score
                            
                        # Look for a significant spike (adjust 0.015 based on your model's variance)
                        if current_score - baseline_mineclip_score > 0.015 or current_score>0.25:
                            print(f"\n[MineCLIP] Success! Score spiked from {baseline_mineclip_score:.4f} to {current_score:.4f} for '{current_prompt}'")
                            task_completed = True
                            break
                    
                    ticks += 1
                    
            # 4. Update status for the next Mistral query
            current_inventory = extract_inventory(obs)
            inventory_str = ", ".join([f"{count}x {item}" for item, count in current_inventory.items()]) if current_inventory else "Empty"
            
            base_status = f"Successfully completed step: '{current_prompt}'." if task_completed else f"Timed out trying to complete: '{current_prompt}'. Agent might be stuck."
            status = f"{base_status}\nCurrent Inventory: {inventory_str}"
                
    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopping execution and saving video data...")
        
    finally:
        # Cleanup and Save
        cv2.destroyAllWindows()
        if video_frames:
            os.makedirs(output_video_dirpath, exist_ok=True)
            safe_goal_name = "".join(x for x in high_level_goal if x.isalnum() or x in " -_").replace(" ", "_")
            video_filepath = os.path.join(output_video_dirpath, f'autonomous_{safe_goal_name}.mp4')
            json_filepath = os.path.join(output_video_dirpath, f'autonomous_{safe_goal_name}.json')
            
            print(f'Saving video to {video_filepath}...')
            save_frames_as_video(video_frames, video_filepath, fps=FPS)
            
            print(f'Saving prompts JSON to {json_filepath}...')
            with open(json_filepath, 'w') as f:
                json.dump(frame_prompts, f)
            print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_model', type=str, default='data/weights/vpt/2x.model')
    parser.add_argument('--in_weights', type=str, default='data/weights/steve1/steve1.weights')
    parser.add_argument('--prior_weights', type=str, default='data/weights/steve1/steve1_prior.pt')
    parser.add_argument('--output_video_dirpath', type=str, default='data/generated_videos/autonomous_videos')
    parser.add_argument('--minecraft_seed', type=float, default=None)
    parser.add_argument('--cond_scale', type=float, default=6.0)
    args = parser.parse_args()

    agent, mineclip, env = load_mineclip_agent_env(args.in_model, args.in_weights, args.minecraft_seed, args.cond_scale)
    prior = load_vae_model(PRIOR_INFO)
    print(mineclip.image_encoder.pos_embed.shape)
    goal = input("Enter your high-level goal (e.g., 'Craft a wooden pickaxe'):\n>")
    run_autonomous_mineclip(
        agent=agent, 
        mineclip=mineclip, 
        prior=prior, 
        env=env, 
        high_level_goal=goal, 
        device=DEVICE, 
        output_video_dirpath=args.output_video_dirpath
    )