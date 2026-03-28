import argparse
import json
import os

import cv2
import numpy as np
import torch
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


import json
import requests

def get_next_step_mistral(high_level_goal, current_task_status):
    """Queries a local Mistral model via Ollama to get the next step."""
    
    system_prompt = """
    You are a Minecraft planning agent. The user wants to achieve a high-level goal.
    Based on the current status, output the NEXT single action STEVE-1 should take.
    You must output valid JSON with two keys:
    1. "steve_prompt": A short text command like "chop a tree", "build a tower", or "get dirt".
    2. "rationale": A brief explanation of why this step is necessary.
    """
    
    prompt = f"Goal: {high_level_goal}\nStatus: {current_task_status}"
    
    # Using local Ollama API
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral",
        "prompt": f"{system_prompt}\n\n{prompt}",
        "format": "json",
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response_data = response.json()
        return json.loads(response_data["response"])
    except Exception as e:
        print(f"Error querying Mistral: {e}")
        return {"steve_prompt": "explore", "rationale": "Fallback due to error."}

def run_interactive(in_model, in_weights, cond_scale, seed, prior_info, output_video_dirpath):
    """Runs the agent in the MineRL env and allows the user to enter prompts to control the agent.
    Clicking on the gameplay window will pause the gameplay and allow the user to enter a new prompt.

    Typing 'reset agent' will reset the agent's state.
    Typing 'reset env' will reset the environment.
    Typing 'save video' will save the video so far (and ask for a video name). It will also save a json storing
        the active prompt at each frame of the video.
    """
    agent, mineclip, env = load_mineclip_agent_env(in_model, in_weights, seed, cond_scale)
    prior = load_vae_model(prior_info)
    window_name = 'STEVE-1 Gameplay (Click to Enter Prompt)'

    state = {'obs': None}
    os.makedirs(output_video_dirpath, exist_ok=True)
    video_frames = []
    frame_prompts = []

    def handle_prompt():
        # Pause the gameplay and ask for a new prompt
        prompt = input('\n\nEnter a prompt:\n>').strip().lower()

        # Reset the agent or env if prompted
        if prompt == 'reset agent':
            print('\n\nResetting agent...')
            agent.reset(cond_scale)
            print(f'Done. Continuing gameplay with previous prompt...')
            return
        elif prompt == 'reset env':
            reset_env()
            print(f'Done. Continuing gameplay with previous prompt...')
            return

        # Save the video so far if prompted
        if prompt == 'save video':
            # Ask for a video name
            video_name = input('Enter a video name:\n>').strip().lower()

            # Save both the video and the prompts for each frame
            output_video_filepath = os.path.join(output_video_dirpath, f'{video_name}.mp4')
            prompts_for_frames_filepath = os.path.join(output_video_dirpath, f'{video_name}.json')
            print(f'Saving video to {output_video_filepath}...')
            save_frames_as_video(video_frames, output_video_filepath, fps=FPS)
            print(f'Saving prompts for frames to {prompts_for_frames_filepath}...')
            with open(prompts_for_frames_filepath, 'w') as f:
                json.dump(frame_prompts, f)
            print(f'Done. Continuing gameplay with previous prompt...')
            return

        # Use prior to get the prompt embed
        prompt_embed = get_prior_embed(prompt, mineclip, prior, DEVICE)

        with torch.cuda.amp.autocast():
            while True:
                minerl_action = agent.get_action(state['obs'], prompt_embed)
                state['obs'], _, _, _ = env.step(minerl_action)

                frame = create_video_frame(state['obs']['pov'], prompt)
                video_frames.append(frame)
                frame_prompts.append(prompt)
                cv2.imshow(window_name, frame)
                cv2.waitKey(1)

    def reset_env():
        print('\nResetting environment...')
        state['obs'] = env.reset()
        if seed is not None:
            print(f'Setting seed to {seed}...')
            env.seed(seed)
    reset_env()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            handle_prompt()

    initial_frame = create_video_frame(state['obs']['pov'], 'Click to Enter a Prompt')
    cv2.imshow(window_name, initial_frame)
    cv2.setMouseCallback(window_name, on_click)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Close the window when 'q' is pressed
            break

import cv2
from collections import deque
from tqdm import tqdm
from steve1.utils.embed_utils import get_prior_embed

def run_autonomous_mineclip(agent, mineclip, prior, env, high_level_goal, device):
    """Runs the agent using Mistral for planning and MineCLIP for success criteria."""
    
    # Make sure seed is set if specified
    obs = env.reset()[cite: 2]
    status = "Just spawned in a new world. No items yet."
    
    # MineCLIP requires a 16-frame video snippet for embedding
    frame_buffer = deque(maxlen=16)
    
    while True:
        # 1. Ask Mistral for the next step
        print(f"\n[MISTRAL] Planning next step for goal: {high_level_goal}...")
        plan = get_next_step_mistral(high_level_goal, status)
        current_prompt = plan.get("steve_prompt", "explore")
        
        print(f"[PLANNER] Rationale: {plan.get('rationale', '')}")
        print(f"[PLANNER] Executing Step: '{current_prompt}'")
        
        # 2. Embed the prompt using STEVE-1's prior
        prompt_embed = get_prior_embed(current_prompt, mineclip, prior, device)
        
        # 3. Execution loop for the current step
        ticks = 0
        max_ticks = 1000 # 50 seconds at 20 FPS
        task_completed = False
        
        print(f"Running '{current_prompt}'...")
        with torch.cuda.amp.autocast():
            while ticks < max_ticks:
                # Get action and step env
                minerl_action = agent.get_action(obs, prompt_embed)[cite: 2]
                obs, _, _, _ = env.step(minerl_action)[cite: 2]
                
                # Process frame for video buffer
                frame = obs['pov'][cite: 2]
                frame_resized = cv2.resize(frame, (128, 128))[cite: 2]
                # Convert to tensor and normalize (adapt based on specific MineCLIP pre-processing)
                frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
                frame_buffer.append(frame_tensor)
                
                # Check success every 16 frames to save compute
                if ticks % 16 == 0 and len(frame_buffer) == 16:
                    # You will need to tune this threshold!
                    if check_mineclip_success(mineclip, frame_buffer, current_prompt, device, threshold=0.28):
                        print(f"[MineCLIP] Success threshold met for '{current_prompt}'!")
                        task_completed = True
                        break
                
                ticks += 1
                
        # 4. Update status for the next Mistral query
        if task_completed:
            status = f"Successfully completed step: '{current_prompt}'."
        else:
            status = f"Timed out trying to complete: '{current_prompt}'. Agent might be stuck."


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_model', type=str, default='data/weights/vpt/2x.model')
    parser.add_argument('--in_weights', type=str, default='data/weights/steve1/steve1.weights')
    parser.add_argument('--prior_weights', type=str, default='data/weights/steve1/steve1_prior.pt')
    parser.add_argument('--output_video_dirpath', type=str, default='data/generated_videos/interactive_videos')
    parser.add_argument('--minecraft_seed', type=float, default=None)  # None for random seed
    parser.add_argument('--cond_scale', type=float, default=6.0)
    args = parser.parse_args()

    agent, mineclip, env = load_mineclip_agent_env(args.in_model, args.in_weights, args.minecraft_seed, args.cond_scale)
    prior = load_vae_model(PRIOR_INFO)

    goal = input("Enter your high-level goal (e.g., 'Craft a wooden pickaxe'):\n>")
    run_autonomous_mineclip(agent, mineclip, prior, env, goal)