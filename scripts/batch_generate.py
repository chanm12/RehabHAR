
import os
import sys
import argparse
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to path to ensure modules are found
sys.path.append(os.getcwd())

from prompt_generation.prompt import generate_promt, generate_visual_prompt, generate_label_semantics, generate_vq_prompt, _get_device_location
from prompt_generation.chart import generate_imu_chart
from prompt_generation.eval import evaluate_answer

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

def call_llm(client, model, messages):
    """
    Calls OpenAI or Anthropic API depending on the model name.
    """
    is_anthropic = model.startswith("claude-")
    
    if is_anthropic:
        # Extract system message, as Anthropic passes it separately
        system_msg = ""
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                # Handle text vs visual
                if isinstance(msg["content"], list):
                    anth_content = []
                    for item in msg["content"]:
                        if item["type"] == "text":
                            anth_content.append({"type": "text", "text": item["text"]})
                        elif item["type"] == "image_url":
                            # Translate OpenAI data URI to Anthropic base64 source block
                            b64_data = item["image_url"]["url"].split("base64,")[1]
                            anth_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": b64_data
                                }
                            })
                    anthropic_messages.append({"role": msg["role"], "content": anth_content})
                else:
                    anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
                    
        resp = client.messages.create(
            model=model,
            system=system_msg,
            messages=anthropic_messages,
            max_tokens=2048,
            temperature=0.5,
            timeout=30.0
        )
        return resp.content[0].text

    # Default to OpenAI logic
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5, # Default temperature
            timeout=30.0
        )
        return resp.choices[0].message.content
    except Exception as e:
        # Retry with temperature=1.0 if unsupported value error (common for o1/o3)
        if "temperature" in str(e) and ("1" in str(e) or "default" in str(e)):
            print(f"  [info] Retrying with temperature=1.0 due to model constraint...")
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.0,
                timeout=30.0
            )
            return resp.choices[0].message.content
        else:
            raise e

def main():
    parser = argparse.ArgumentParser(description="Batch process IMU data to generate prompts and LLM responses.")
    parser.add_argument("--npy", default="data/uci_data.npy", help="Path to NumPy array file (N,T,6).")
    parser.add_argument("--dataset", default="uci", help="Dataset name.")
    parser.add_argument("--exp_id", default="baseline", help="Experiment ID for grouping outputs (e.g. A1, E1, S1).")
    parser.add_argument("--output_dir", default="output/batch_results", help="Directory to save outputs.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index.")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to process (default: all).")
    parser.add_argument("--call", action="store_true", help="Call LLM for each sample.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest directory and skip already generated prompts.")
    parser.add_argument("--mode", choices=["analysis", "visual", "vq"], default="analysis", help="Mode: analysis, visual, or vq (discrete token sequence).")
    parser.add_argument("--quantized", default=None, help="Path to quantized .npy file (required for --mode vq).")
    parser.add_argument("--vq_K", type=int, default=64, help="Codebook size K used during VQ training (for prompt context).")
    parser.add_argument("--sensors", default=None, help="Path to sensors .npy integer array (optional, for multi-sensor datasets).")
    args = parser.parse_args()

    # Parse arguments
    # (Output directory setup is moved until after model name resolution)

    # Load VQ quantized data (if VQ mode)
    quantized_data = None
    if args.mode == "vq":
        if not args.quantized:
            print("Error: --mode vq requires --quantized <path to quantized .npy>")
            return
        print(f"Loading quantized data from {args.quantized}...")
        quantized_data = np.load(args.quantized, allow_pickle=True)
        print(f"  Quantized trials: {len(quantized_data)}")

    # Load Sensors Data (if provided)
    sensors_data = None
    if args.sensors:
        print(f"Loading sensors data from {args.sensors}...")
        try:
            sensors_data = np.load(args.sensors, allow_pickle=True)
        except Exception as e:
            print(f"Error loading sensors data: {e}")
            return

    # Load raw IMU Data (not needed for pure VQ mode, but load anyway for metadata)
    print(f"Loading data from {args.npy}...")
    try:
        data = np.load(args.npy, allow_pickle=True)
        if data.shape == () and data.dtype == object:
            data = data.item()
        if isinstance(data, list):
            data = np.array(data, dtype=object)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Data shape: {data.shape}")
    
    # Setup LLM
    client = None
    model_name = "gpt-4o-mini" # Default
    if args.call:
        load_dotenv()
        model_name = os.getenv("MODEL") or model_name
        
        if model_name.startswith("claude-"):
            if Anthropic is None:
                print("Error: anthropic SDK not installed. Please pip install anthropic.")
                return
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("Error: ANTHROPIC_API_KEY not found in .env. Cannot call Anthropic LLM.")
                return
            client = Anthropic(api_key=api_key)
        else:
            if OpenAI is None:
                print("Error: openai SDK not installed. Please pip install openai.")
                return
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Error: OPENAI_API_KEY not found in .env. Cannot call OpenAI LLM.")
                return
            client = OpenAI(api_key=api_key)
            
        print(f"LLM initialized with model: {model_name}")

    # Determine Range
    total_samples = data.shape[0]
    end_index = total_samples
    if args.limit:
        end_index = min(total_samples, args.start_index + args.limit)
    # Resolve effective model for output path
    effective_model = os.getenv("VLM_MODEL") or "gpt-4o" if args.mode == "visual" else model_name
    
    # Setup structured output directory by Experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{args.dataset}_{args.mode}_{effective_model}"
    base_dir = os.path.join("output", "experiments", args.exp_id, folder_name)
    
    experiment_dir = os.path.join(base_dir, timestamp)
    if args.resume and os.path.exists(base_dir):
        subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if subdirs:
            experiment_dir = max(subdirs, key=os.path.getmtime)
            print(f"Resuming from latest directory: {experiment_dir}")

    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save Run Configuration
    config = vars(args).copy()
    config["timestamp"] = timestamp
    config["effective_model"] = effective_model
    config["total_samples_available"] = total_samples
    config["processed_samples_range"] = [args.start_index, end_index]
    
    with open(os.path.join(experiment_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"Outputs will be saved directly to: {experiment_dir}")
    print(f"Processing samples {args.start_index} to {end_index}...")

    success_count = 0

    for i in tqdm(range(args.start_index, end_index)):
        try:
            # Prepare Sample
            # Handle object array (variable length) vs fixed array
            if hasattr(data, 'dtype') and data.dtype == object:
                sample = data[i]
                acc = sample[:, :3].astype(float)
                gyro = sample[:, 3:6].astype(float)
            else:
                acc = data[i, :, :3].astype(float)
                gyro = data[i, :, 3:6].astype(float)
                
            # Determine Sensor ID if variable
            sensor_idx = int(sensors_data[i]) if sensors_data is not None else None
            
            # Paths (using zero-padded indices)
            base_filename = f"{i:05d}"
            prompt_path = os.path.join(experiment_dir, f"{base_filename}_prompt.txt")
            result_path = os.path.join(experiment_dir, f"{base_filename}_result.txt")
            
            # Skip if already generated and valid
            if args.resume and os.path.exists(result_path) and os.path.getsize(result_path) > 10:
                success_count += 1
                continue
            
            # Generate Prompt
            if args.mode == "visual":
                chart_path = os.path.join(experiment_dir, f"{base_filename}_chart.png")

                # Get human readable device string for chart title
                device_str = _get_device_location(args.dataset, sensor_idx, label=None)
                chart_title = f"{args.dataset.upper()} Sample {i} | Sensor: {device_str.title()}"
                
                chart_b64 = generate_imu_chart(acc, gyro, save_path=chart_path, title=chart_title)
                prompt = generate_visual_prompt(args.dataset, sensor_idx=sensor_idx, label=None)
                messages = [
                    {"role": "system", "content": "You are a senior motion analysis expert."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{chart_b64}"}}
                    ]}
                ]
                active_model = os.getenv("VLM_MODEL") or "gpt-4o"
                
                # If VLM is Claude, we need the Anthropic client initialized 
                # (in case the base text model was OpenAI but VLM is Claude)
                if active_model.startswith("claude-") and not isinstance(client, Anthropic):
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                    if not api_key:
                        print(f"Failed on sample {i}: ANTHROPIC_API_KEY required for {active_model}")
                        continue
                    vlm_client = Anthropic(api_key=api_key)
                else:
                    vlm_client = client

            elif args.mode == "vq":
                token_sequence = quantized_data[i].tolist() if hasattr(quantized_data[i], "tolist") else list(quantized_data[i])
                prompt = generate_vq_prompt(token_sequence, args.dataset, K=args.vq_K, sensor_idx=sensor_idx, label=None)
                messages = [
                    {"role": "system", "content": "You are a senior motion analysis expert specializing in movement pattern analysis."},
                    {"role": "user", "content": prompt}
                ]
                active_model = model_name

            else:  # analysis mode
                prompt = generate_promt(acc, gyro, args.dataset, label=None, sensor_idx=sensor_idx)
                messages = [
                    {"role": "system", "content": "You are a senior motion analysis expert."},
                    {"role": "user", "content": prompt}
                ]
                active_model = model_name

            # Save Prompt
            with open(prompt_path, "w") as f:
                f.write(prompt)
            
            # Call LLM
            if args.call:
                active_client = vlm_client if args.mode == "visual" else client
                response_text = call_llm(active_client, active_model, messages)
                with open(result_path, "w") as f:
                    f.write(response_text)
            
            success_count += 1
            
        except Exception as e:
            print(f"Failed on sample {i}: {e}")

    print(f"Batch processing complete. Processed {success_count} samples.")
    print(f"Outputs saved to: {experiment_dir}")

if __name__ == "__main__":
    main()
