"""Replace the website placeholders with website domains from env_config
Generate the test data"""
import json
import os
import argparse
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from browser_env.env_config import *


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True, 
                       help="Environment name (e.g., 180_209_7_95, webarena1, webarena2)")
    parser.add_argument("--raw_data", type=str, required=True,
                       help="Path to raw data JSON file (default: config_files/webarena/test.raw.json)")
    parser.add_argument("--output_prefix", type=str, required=True,
                       help="Output file prefix (default: test). Output will be {output_prefix}.json")
    args = parser.parse_args()
    
    with open(args.raw_data, "r") as f:
        raw = f.read()
    
    # Replace website placeholders
    raw = raw.replace("__GITLAB__", GITLAB)
    raw = raw.replace("__REDDIT__", REDDIT)
    raw = raw.replace("__SHOPPING__", SHOPPING)
    raw = raw.replace("__SHOPPING_ADMIN__", SHOPPING_ADMIN)
    raw = raw.replace("__WIKIPEDIA__", WIKIPEDIA)
    raw = raw.replace("__MAP__", MAP)
    
    # Replace storage_state paths with environment-specific auth cookie paths
    raw = re.sub(r'"storage_state": "\.\/\.auth\/(.*?)"', 
                f'"storage_state": "./.auth/{args.env_name}_{args.output_prefix}/\\1"', raw)
    
    # Extract file name from full path and remove .raw suffix
    raw_filename = os.path.basename(args.raw_data)  # Keep only the file name
    output_prefix = raw_filename.replace(".raw.json", ".json")  # Remove .raw suffix
    
    output_file = f"config_files/webarena/{output_prefix}"
    with open(output_file, "w") as f:
        f.write(raw)

    tasks_dir = f"config_files/webarena/{args.env_name}_{args.output_prefix}"
    os.makedirs(tasks_dir, exist_ok=True)
    
    # split to multiple files
    data = json.loads(raw)
    for idx, item in enumerate(data):
        file_name = item["task_id"]
        with open(f"{tasks_dir}/{file_name}.json", "w") as f:
            json.dump(item, f, indent=2)


if __name__ == "__main__":
    main()
