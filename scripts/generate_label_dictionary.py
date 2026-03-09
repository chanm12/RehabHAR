import os
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

from prompt_generation.prompt import _DATASET_ACTIVITIES, generate_label_semantics

# Import provider clients
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None

def get_unique_activities():
    """Extract and clean all unique activities from the dataset lists."""
    all_activities = set()
    for ds, act_str in _DATASET_ACTIVITIES.items():
        # Clean the string e.g., "[walking, biking]" -> "walking", "biking"
        clean_str = act_str.strip('[]')
        acts = [a.strip() for a in clean_str.split(',')]
        for a in acts:
            if a and a != 'unknown':
                all_activities.add(a.lower())
    
    # Also ensure "still" is in there since read_data.py maps sit/stand to still
    all_activities.add("still")
    all_activities.add("stairsup")
    all_activities.add("stairsdown")
    
    return sorted(list(all_activities))


def call_llm(client, model: str, prompt: str) -> str:
    """Helper to call LLM for just a simple text prompt."""
    is_anthropic = model.startswith("claude-")
    
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    
    if is_anthropic:
        resp = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0.2, # Low temp for deterministic definitions
            messages=messages
        )
        return resp.content[0].text
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive Dual Semantics dictionary for all labels.")
    parser.add_argument("--model", default=None, help="Model to use (default: env MODEL or gpt-4o-mini)")
    parser.add_argument("--output", default="data/label_semantics.json", help="Output JSON path")
    args = parser.parse_args()

    load_dotenv()
    
    model = args.model or os.getenv("MODEL") or "gpt-4o-mini"
    
    # Initialize client
    if model.startswith("claude-"):
        if not Anthropic:
            raise RuntimeError("Anthropic SDK not installed.")
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    else:
        if not OpenAI:
            raise RuntimeError("OpenAI SDK not installed.")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    activities = get_unique_activities()
    print(f"Discovered {len(activities)} unique activities across all datasets.")
    
    dictionary = {}
    
    # Load existing to resume if needed
    if os.path.exists(args.output):
        print(f"Loading existing dictionary from {args.output} to resume...")
        with open(args.output, 'r') as f:
            dictionary = json.load(f)

    for act in tqdm(activities, desc="Generating Semantics"):
        if act not in dictionary:
            dictionary[act] = {}
            
        if "healthy" not in dictionary[act]:
            prompt_healthy = generate_label_semantics(act, is_impaired=False)
            try:
                res_healthy = call_llm(client, model, prompt_healthy)
                dictionary[act]["healthy"] = res_healthy
            except Exception as e:
                print(f"Error generating healthy semantics for {act}: {e}")
                
        if "impaired" not in dictionary[act]:
            prompt_impaired = generate_label_semantics(act, is_impaired=True)
            try:
                res_impaired = call_llm(client, model, prompt_impaired)
                dictionary[act]["impaired"] = res_impaired
            except Exception as e:
                print(f"Error generating impaired semantics for {act}: {e}")
                
        # Save incrementally
        with open(args.output, 'w') as f:
            json.dump(dictionary, f, indent=4)

    print(f"Dictionary generation complete! Saved to {args.output}")

if __name__ == "__main__":
    main()
