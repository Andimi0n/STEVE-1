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
from steve1.utils.state import extract_inventory

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


import openai # or your preferred LLM SDK
import json

def get_next_step_from_llm(high_level_goal, current_inventory):
    """Asks the LLM what STEVE-1 should do next based on the state."""
    
    system_prompt = """
    You are a Minecraft planning agent. The user wants to achieve a high-level goal.
    Based on the current inventory, output the NEXT single action STEVE-1 should take.
    You must output valid JSON with two keys:
    1. "steve_prompt": A short text command like "chop a tree" or "get dirt".
    2. "target_item": The item we are trying to get in this step (e.g., "log").
    3. "target_count": The minimum amount of that item needed to move on.
    """
    
    user_prompt = f"Goal: {high_level_goal}\nCurrent Inventory: {current_inventory}"
    
    # Example OpenAI call (replace with your actual LLM call)
    response = openai.ChatCompletion.create(
        model="gpt-4", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # Parse the JSON response
    return json.loads(response.choices[0].message.content)

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

def run_autonomous(agent, mineclip, prior, env, high_level_goal):
    """Runs the agent autonomously using an LLM planner."""
    state = {'obs': env.reset()}
    
    while True:
        # 1. Get current state
        current_inv = extract_inventory(state['obs'])
        print(f"Current Inventory: {current_inv}")
        
        # 2. Ask LLM for the next step
        plan = get_next_step_from_llm(high_level_goal, current_inv)
        current_prompt = plan["steve_prompt"]
        target_item = plan["target_item"]
        target_count = plan["target_count"]
        
        print(f"\n[PLANNER] Executing: '{current_prompt}' to get {target_count} {target_item}")
        
        # 3. Embed the prompt (Using STEVE-1's prior)
        prompt_embed = get_prior_embed(current_prompt, mineclip, prior, DEVICE)
        
        # 4. Execute the prompt until the condition is met (or timeout)
        ticks = 0
        max_ticks = 1200 # 60 seconds at 20 FPS to prevent infinite loops
        
        with torch.cuda.amp.autocast():
            while ticks < max_ticks:
                minerl_action = agent.get_action(state['obs'], prompt_embed)
                state['obs'], _, done, _ = env.step(minerl_action)
                
                # Check if we got the item
                current_inv = extract_inventory(state['obs'])
                if current_inv.get(target_item, 0) >= target_count:
                    print(f"[PLANNER] Success! Got {target_count} {target_item}.")
                    break # Break out to ask the LLM for the next step
                
                # Render video frame (optional, adapted from your code)
                frame = create_video_frame(state['obs']['pov'], current_prompt)
                cv2.imshow('STEVE-1 Autonomous Planner', frame)
                cv2.waitKey(1)
                
                ticks += 1
                
            if ticks >= max_ticks:
                print(f"[PLANNER] Timeout reached for '{current_prompt}'. Re-planning...")


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
    run_autonomous(agent, mineclip, prior, env, goal)