import ollama
import sys
import argparse
import json
import re
import subprocess
import shutil
import fnmatch
import time
import os  # <--- Added this to handle path names
from typing import List, Optional, Tuple, Any, Dict
from textwrap import dedent

try:
    from pydantic import ValidationError
except ImportError:
    class ValidationError(Exception): pass

# --- Helpers ---

class Logger:
    """Simple logger to handle CLI verbosity vs Library silence."""
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def info(self, msg: str, end: str = "\n", flush: bool = False):
        if self.verbose:
            print(msg, end=end, flush=flush, file=sys.stdout)

    def error(self, msg: str):
        print(f"[ERR] {msg}", file=sys.stderr)

    def warn(self, msg: str):
        if self.verbose:
            print(f"[WARN] {msg}", file=sys.stderr)

# Global logger instance (default to silent for library use, enabled in main)
log = Logger(verbose=False)

def _require_nvidia() -> bool:
    """Checks if nvidia-smi is available."""
    return shutil.which('nvidia-smi') is not None

# --- Library Functions ---

def get_running_models() -> List[Any]:
    try:
        # Note: ollama.ps() returns a specific object structure
        current_status = ollama.ps()
        return current_status.models
    except Exception as e:
        log.error(f"Retrieving running status: {e}")
        return []

def get_available_models(sort_by: str = 'name', reverse: bool = False) -> List[Any]:
    try:
        response = ollama.list()
        models = response.models
        
        if sort_by == 'size':
            key_func = lambda x: x.size
        else:
            key_func = lambda x: x.model.lower()
            
        models.sort(key=key_func, reverse=reverse)
        return models
    except Exception as e:
        log.error(f"Retrieving local models: {e}")
        return []

def stop_models() -> List[str]:
    stopped_models = []
    try:
        running_models = get_running_models()
        if not running_models: 
            return []
            
        for model_obj in running_models:
            name = model_obj.name
            # keep_alive=0 forces immediate unload
            ollama.generate(model=name, prompt="", keep_alive=0)
            stopped_models.append(name)
    except Exception as e:
        log.error(f"Stopping models: {e}")
    return stopped_models

def load_models(model_names: List[str]):
    if not model_names: return
    log.info(f"Loading models: {', '.join(model_names)}")
    
    for name in model_names:
        log.info(f"  > Pinging {name}...", end=" ", flush=True)
        try:
            ollama.generate(model=name, prompt="") 
            log.info("OK.")
        except Exception as e:
            log.info(f"Failed: {e}")

def delete_models_interactive(pattern: str, skip_confirm: bool = False):
    """
    Batch delete models matching a pattern with interactive confirmation.
    """
    # 1. Get Inventory
    try:
        # We need raw names, get_available_models returns objects
        all_models = [m.model for m in get_available_models()]
    except Exception as e:
        log.error(f"Error listing models: {e}")
        return

    # 2. Filter
    matches = fnmatch.filter(all_models, pattern)
    
    if not matches:
        log.info(f"No models found matching pattern: '{pattern}'")
        return

    # 3. Confirmation
    log.info(f"Found {len(matches)} models matching '{pattern}':")
    for m in matches:
        log.info(f"  - {m}")
    
    if not skip_confirm:
        confirm = input(f"\nAre you sure you want to PERMANENTLY delete these {len(matches)} models? [y/N]: ")
        if confirm.lower() not in ['y', 'yes']:
            log.info("Aborted.")
            return

    # 4. Execution
    log.info("\nDeleting...")
    for m in matches:
        log.info(f"  Running 'ollama rm {m}'...", end=" ", flush=True)
        try:
            ollama.delete(m)
            log.info("OK")
        except Exception as e:
            log.info(f"FAILED: {e}")

    log.info("Done.")

def parse_size_input(size_str: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Parses '32k' (tokens) or '100b' (bytes) into (bytes_int, label_str).
    Returns (None, None) on failure.
    """
    s = size_str.strip().lower()
    try:
        if s.endswith('b'):
            # Explicit bytes
            val_bytes = int(s[:-1])
            k_val = val_bytes / 1024
        elif s.endswith('k'):
            # Kilo-tokens (Context size)
            k_val = float(s[:-1])
            val_bytes = int(k_val * 1024)
        else:
            k_val = float(s)
            val_bytes = int(k_val * 1024)
            
        label = f"{int(k_val)}" if k_val.is_integer() else f"{k_val:.2f}"
        return val_bytes, label
    except ValueError:
        return None, None

def create_context_variant(base_model: str, ctx_bytes: int, ctx_label: str) -> Optional[str]:
    clean_name = re.sub(r'-ctx[\d\.]+[kK]$', '', base_model)
    new_model_name = f"{clean_name}-ctx{ctx_label}K"
    
    log.info(f"Creating new model: {new_model_name}")
    log.info(f"  > Base: {base_model}")
    log.info(f"  > Context: {ctx_label}K ({ctx_bytes} bytes)")
    
    try:
        ollama.create(model=new_model_name, from_=base_model, parameters={'num_ctx': ctx_bytes})
        log.info(f"Success! Created {new_model_name}")
        return new_model_name
    except Exception as e:
        log.error(f"Creating model variant: {e}")
        return None

# --- Optimization Logic ---

def get_gpu_free_memory() -> Optional[int]:
    """Returns free VRAM in Bytes using nvidia-smi. Returns None if unavailable."""
    if not _require_nvidia(): 
        return None
    try:
        # Use --id=0 for multi-gpu? Currently grabs the first one outputted.
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        lines = output.strip().split('\n')
        # Take the first GPU found
        return int(lines[0]) * 1024 * 1024
    except Exception:
        return None

def find_val_fuzzy(info_dict: Dict, candidates: List[str]) -> Optional[Any]:
    cand_set = set(candidates)
    for key, value in info_dict.items():
        if key in cand_set: return value
        # Handle nested keys like 'attention.head_count'
        if '.' in key:
            suffix = key.split('.')[-1]
            if suffix in cand_set: return value
    return None

def scrape_cli_metadata(model_name: str) -> Dict[str, Any]:
    if not shutil.which('ollama'): return {}
    try:
        result = subprocess.run(['ollama', 'show', model_name], capture_output=True, text=True)
        if result.returncode != 0: return {}
        
        data = {}
        # Regex to capture "Key   Value" pairs allowing for units
        # e.g. "context length   8192" or "parameters   8.0B"
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line: continue
            
            match = re.search(r'^(.*?)\s+([\d\.]+[KMG]?B?)$', line, re.IGNORECASE)
            if match:
                raw_key = match.group(1).strip().replace(" ", "_").lower()
                val_str = match.group(2).upper()
                
                # Special case for parameter count (8.0B)
                if raw_key == 'parameters' and val_str.endswith('B'):
                     try:
                        data['parameters_b'] = float(val_str[:-1])
                     except: pass
                     continue

                # Parse units
                mult = 1
                if val_str.endswith('K'): mult = 1024; val_str = val_str[:-1]
                elif val_str.endswith('M'): mult = 1024**2; val_str = val_str[:-1]
                elif val_str.endswith('G'): mult = 1024**3; val_str = val_str[:-1]
                
                try:
                    data[raw_key] = int(float(val_str) * mult)
                except ValueError:
                    continue
        return data
    except Exception:
        return {}

def estimate_architecture_from_params(params_b: float) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Returns (layers, embed_dim, head_count, head_count_kv) based on parameter count.
    Used as fallback when metadata is missing.
    """
    if not params_b: return None, None, None, None
    
    # Heuristics based on common architectures (Llama 3, Mistral, Qwen)
    
    # < 4B: usually small models (Phi, Qwen-1.8) -> Assume MHA
    if params_b < 4.0:
        return 32, 2560, 32, 32
        
    # 7B - 9B (Llama 3 8B, Qwen 7B, Mistral 7B)
    elif 4.0 <= params_b < 10.0:
        return 32, 4096, 32, 8
        
    # 10B - 16B (Qwen 14B, DeepSeek 16B)
    elif 10.0 <= params_b < 18.0:
        return 40, 5120, 40, 10
        
    # 30B - 35B (Qwen 32B, Command R)
    elif 18.0 <= params_b < 40.0:
        return 60, 6656, 40, 8
        
    # 70B+ (Llama 3 70B)
    elif 40.0 <= params_b:
        return 80, 8192, 64, 8
        
    return None, None, None, None

def benchmark_model_memory(model_name: str, debug: bool = False) -> Tuple[Optional[float], Optional[int]]:
    """
    Loads model at 2k and 8k context using a REAL PROMPT to force allocation.
    Returns (bytes_per_token, baseline_model_weights).
    """
    def wait_for_unload(target_model, timeout=10):
        start = time.time()
        while time.time() - start < timeout:
            running = get_running_models()
            is_present = any(m.name == target_model or m.model == target_model for m in running)
            if not is_present: return True
            time.sleep(0.5)
        return False

    def force_unload(target_model):
        try:
            ollama.generate(model=target_model, prompt="", keep_alive=0)
            wait_for_unload(target_model)
            time.sleep(0.5)
            return True
        except:
            return False

    try:
        force_unload(model_name)
        base_free = get_gpu_free_memory()
        
        if debug: log.info(f"       DEBUG: Loading 2K...", end=" ", flush=True)
        ollama.generate(model=model_name, prompt="Why is the sky blue?", options={"num_ctx": 2048, "num_predict": 1})
        time.sleep(0.5) 
        free_2k = get_gpu_free_memory()
        used_2k = base_free - free_2k
        if debug: log.info(f"Used: {format_size(used_2k)}")
        
        force_unload(model_name)

        if debug: log.info(f"       DEBUG: Loading 8K...", end=" ", flush=True)
        ollama.generate(model=model_name, prompt="Why is the sky blue?", options={"num_ctx": 8192, "num_predict": 1})
        time.sleep(0.5)
        free_8k = get_gpu_free_memory()
        used_8k = base_free - free_8k
        if debug: log.info(f"Used: {format_size(used_8k)}")

        force_unload(model_name)

        delta_bytes = used_8k - used_2k
        delta_tokens = 8192 - 2048
        
        if delta_bytes <= 0:
            if debug: log.info(f"       DEBUG: FAILED (No growth: {delta_bytes})")
            return None, None

        bytes_per_token = delta_bytes / delta_tokens
        weights_approx = used_2k - (2048 * bytes_per_token)
        return bytes_per_token, weights_approx

    except Exception as e:
        if debug: log.info(f"       DEBUG: Benchmark exception: {e}")
        return None, None

def calculate_max_context(model_name: str, existing_models: set, buffer_gb: float = 1.0, 
                          batch_mode: bool = False, debug: bool = False, manual_mode: bool = False):
    
    prefix = f"[....] {model_name}" if batch_mode else f"{model_name}"
    
    # 1. Immediate Filter: Skip Cloud/Shim names
    if 'cloud' in model_name.lower():
        if batch_mode: log.info(f"[SKIP] {model_name}: Ignored (Cloud/Shim detected).")
        else: log.info(f"Skipping {model_name}: Cloud model detected.")
        return

    if batch_mode:
        log.info(f"{prefix} Processing...", end="\r")

    # Clean start
    ollama.generate(model=model_name, prompt="", keep_alive=0)
    time.sleep(0.2) 

    free_vram = get_gpu_free_memory()
    if free_vram is None:
        log.error(f"{prefix}: No NVIDIA GPU detected (nvidia-smi failed).")
        return

    try:
        bytes_per_token = 0
        model_size_disk = 0

        # 2. Get Weights Size early to filter tiny stubs
        try:
            list_resp = ollama.list()
            all_models = list_resp.models
            model_size_disk = next((m.size for m in all_models if m.model == model_name), None)
            
            # Filter: If model is < 200MB, it's likely a manifest/shim, not a real model.
            if model_size_disk and model_size_disk < (200 * 1024 * 1024):
                if batch_mode: log.info(f"[SKIP] {model_name}: Ignored (Size < 200MB).")
                else: log.info(f"Skipping {model_name}: Model file too small (< 200MB).")
                return

        except Exception as e:
            pass # Continue if list fails, other checks will catch it

        # 3. Try API for Metadata
        model_info = {}
        try:
            info = ollama.show(model_name)
            model_info = info.model_info if hasattr(info, 'model_info') else info.get('model_info', {})
        except (ValidationError, ollama.ResponseError):
            pass 

        # 4. Try CLI Fallback
        if not model_info:
            model_info = scrape_cli_metadata(model_name)
        
        # Extract Static Params
        layers = int(find_val_fuzzy(model_info, ["block_count", "n_layer", "layers", "layer_count", "num_hidden_layers"]) or 0)
        embed_dim = int(find_val_fuzzy(model_info, ["embedding_length", "n_embd", "hidden_size"]) or 0)
        head_count = int(find_val_fuzzy(model_info, ["attention.head_count", "head_count", "n_head", "num_attention_heads"]) or 0)
        head_count_kv = int(find_val_fuzzy(model_info, ["attention.head_count_kv", "head_count_kv", "n_head_kv", "num_key_value_heads"]) or 0)
        train_limit = int(find_val_fuzzy(model_info, ["context_length", "context_window", "n_ctx"]) or 0)
        params_b = model_info.get('parameters_b', 0)

        is_estimate = False

        # --- STRATEGY SELECTION ---
        
        # A. Manual Override
        if manual_mode:
            if not batch_mode:
                log.info(f"\nManual entry requested for '{model_name}'.")
                try:
                    if not embed_dim: embed_dim = int(input("  > Embedding Length: "))
                    if not head_count: head_count = int(input("  > Head Count: "))
                    if not head_count_kv: 
                        hc_kv_in = input(f"  > KV Head Count (Default: {head_count}): ")
                        head_count_kv = int(hc_kv_in) if hc_kv_in.strip() else head_count
                    if not layers: layers = int(input("  > Layer Count: "))
                    # Calc
                    head_dim = embed_dim // head_count
                    bytes_per_token = 2 * layers * head_count_kv * head_dim * 2
                except ValueError:
                    log.error("Invalid input.")
                    return

        # B. Full Static Metadata
        elif layers and embed_dim and head_count:
            if head_count > 0 and head_count_kv == 0: head_count_kv = head_count
            head_dim = embed_dim // head_count
            bytes_per_token = 2 * layers * head_count_kv * head_dim * 2

        # C. Benchmark (Default if metadata missing)
        else:
            if batch_mode: log.info(f"[TEST] {model_name}: Benchmarking memory usage...", end="\r")
            b_per_tok, b_base = benchmark_model_memory(model_name, debug)
            if b_per_tok and b_per_tok > 0:
                bytes_per_token = b_per_tok
                if b_base: model_size_disk = b_base
                if debug: log.info(f"       DEBUG: Benchmarked BPT: {bytes_per_token:.2f}")

        # D. Heuristic Fallback (If Benchmark Failed)
        if (not bytes_per_token or bytes_per_token <= 0) and params_b > 0 and not manual_mode:
            e_layers, e_dim, e_heads, e_kv = estimate_architecture_from_params(params_b)
            if e_layers:
                layers, embed_dim, head_count, head_count_kv = e_layers, e_dim, e_heads, e_kv
                head_dim = embed_dim // head_count
                bytes_per_token = 2 * layers * head_count_kv * head_dim * 2
                is_estimate = True
                if debug: log.info(f"       DEBUG: Heuristic fallback (GQA-aware, {params_b}B): {bytes_per_token:.2f} BPT")

        # E. Failure
        if not bytes_per_token:
            msg = "Metadata missing"
            if not manual_mode: msg += " & Benchmark failed"
            if params_b > 0: msg += " & Heuristic failed"
            
            if batch_mode: log.info(f"[SKIP] {model_name}: {msg}")
            else: log.info(f"Skipping {model_name}: {msg}")
            return

        # --- The Math ---
        if not model_size_disk:
             log.info(f"[SKIP] {model_name}: Could not determine model size.")
             return

        # BUFFER LOGIC: Tight buffer normally, +1GB safety if using Heuristic
        buffer_bytes = int(buffer_gb * 1024**3)
        if is_estimate:
             buffer_bytes += int(1.0 * 1024**3) # Add 1GB safety tax for estimated models

        available_for_context = free_vram - model_size_disk - buffer_bytes
        
        if available_for_context <= 0:
            if batch_mode: log.info(f"[SKIP] {model_name}: Model too large (Weights > Free VRAM).")
            else: log.info(f"Skipping {model_name}: Model too large.")
            return

        max_tokens = int(available_for_context / bytes_per_token)
        
        final_ctx = max_tokens
        if train_limit > 0 and max_tokens > train_limit:
            final_ctx = train_limit

        recommended_k = int(final_ctx / 1024)
        if recommended_k < 1:
            log.info(f"[SKIP] {model_name}: Max context < 1K.")
            return

        clean_name = re.sub(r'-ctx[\d\.]+[kK]$', '', model_name)
        target_name = f"{clean_name}-ctx{recommended_k}K"
        
        if target_name == model_name:
            log.info(f"[SKIP] {model_name}: Already optimized ({recommended_k}K).")
            return

        if target_name in existing_models:
            log.info(f"[SKIP] {model_name}: Target {target_name} already exists.")
            return

        # --- Execution ---
        if not batch_mode:
            log.info(f"\n--- Optimization Report for {model_name} ---")
            if is_estimate: log.info(f"(Note: Heuristic used. Buffer increased by 1.0GB for safety.)")
            log.info(f"Recommendation: ~{recommended_k}K context")
            
            confirm = input(f"Create '{target_name}'? [Y/n]: ")
            if confirm.lower() not in ['n', 'no']:
                create_context_variant(model_name, recommended_k * 1024, str(recommended_k))
        else:
            log.info(f"[OK]   {model_name}: {recommended_k}K -> {target_name}", end="... ")
            log.info("Creating.")
            create_context_variant(model_name, recommended_k * 1024, str(recommended_k))

    except Exception as e:
        log.error(f"{model_name}: {e}")

# --- CLI Implementation ---

def format_size(s: Optional[int]) -> str: 
    return f"{s/(1024**3):.2f} GB" if s else "0.00 GB"

def main():
    # Enable logging for CLI usage
    log.verbose = True
    
    # Determine the actual program name used to invoke this script
    # e.g., "ollama-ops" or "ollama-ops.py"
    prog_name = os.path.basename(sys.argv[0])

    help_text = dedent(f"""\
        Examples:
          # --- Process Management ---
          {prog_name} stop                     # Stop all running models
          {prog_name} stop --json > s.json     # Stop and save list to file
          {prog_name} load --file s.json       # Restore models from file

          # --- Inventory & Sorting ---
          {prog_name} list                     # List all models (A-Z)
          {prog_name} list --sort size         # List by size (Smallest first)
          {prog_name} list --sort size --desc  # List by size (Largest first)

          # --- Context Management ---
          {prog_name} context llama3 32k       # Manually create a variant (K = tokens)
          {prog_name} context llama3 1048576b  # Byte-precise KV size (B = bytes)
          
          # --- Auto-Optimization ---
          {prog_name} optimize all --buffer 1.5      # Batch optimize ALL models
          {prog_name} optimize "qwen*" --buffer 1.0  # Optimize specific family
          {prog_name} optimize my-model --manual     # Force manual metadata entry

          # --- Cleanup ---
          {prog_name} rm "*-ctx*"              # Delete all models matching pattern
          {prog_name} rm "llama3-ctx*" -y      # Delete without confirmation
        """)

    parser = argparse.ArgumentParser(
        # removed prog='ollama-ops' to allow argparse to use sys.argv[0] by default
        description="Manage local Ollama models: sortable listing, JSON-friendly stop/load, manual variants, auto-optimization, and batch deletion.",
        epilog=help_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command')

    # Subparser definitions
    subparsers.add_parser('stop', help='Stop all running models').add_argument('--json', action='store_true', help='Output JSON')
    
    load_p = subparsers.add_parser('load', help='Load models')
    grp = load_p.add_mutually_exclusive_group(required=False)
    grp.add_argument('--file', type=str, help='Path to JSON file')
    grp.add_argument('--models', type=str, help='Inline JSON list')

    subparsers.add_parser('running', help='Show running models')
    
    list_p = subparsers.add_parser('list', help='List models')
    list_p.add_argument('--sort', default='name', choices=['name', 'size'])
    list_p.add_argument('--desc', action='store_true')

    ctx_p = subparsers.add_parser('context', help='Create manual context variant')
    ctx_p.add_argument('model', type=str)
    ctx_p.add_argument('size', type=str, help='Size (e.g. 32k or 1048576b)')

    opt_p = subparsers.add_parser('optimize', help='Auto-optimize context')
    opt_p.add_argument('model', type=str, help='Model name or pattern')
    opt_p.add_argument('--buffer', type=float, default=1.0, help='VRAM safety buffer (GB)')
    opt_p.add_argument('--debug', action='store_true')
    opt_p.add_argument('--manual', action='store_true')

    rm_p = subparsers.add_parser('rm', help='Batch delete models')
    rm_p.add_argument('pattern', type=str, help='Model name pattern (e.g. "*-ctx*" or "qwen3*")')
    rm_p.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')

    args = parser.parse_args()

    # Pre-flight checks
    if not shutil.which('ollama'):
        log.error("Ollama CLI not found. Is it installed and in PATH?")
        sys.exit(1)

    # Command Routing
    if args.command == 'stop':
        s = stop_models()
        print(json.dumps(s) if args.json else (f"Stopped: {s}" if s else "None running"))
    
    elif args.command == 'load':
        if args.file:
            with open(args.file) as f: load_models(json.load(f))
        elif args.models:
            load_models(json.loads(args.models))
    
    elif args.command == 'running':
        m = get_running_models()
        if m: 
            print(f"{'NAME':<30} {'SIZE (VRAM)':<12}\n" + "-"*42)
            for x in m: print(f"{x.name:<30} {format_size(x.size):<12}")
        else: print("No models loaded.")

    elif args.command == 'list':
        m = get_available_models(args.sort, args.desc)
        if m:
            print(f"{'NAME':<40} {'SIZE':<10} {'FAMILY':<15}\n" + "-"*70)
            for x in m: 
                fam = x.details.family if x.details else "-"
                print(f"{x.model:<40} {format_size(x.size):<10} {fam:<15}")

    elif args.command == 'context':
        b, l = parse_size_input(args.size)
        if b: create_context_variant(args.model, b, l)
        else: log.error("Invalid size format.")

    elif args.command == 'optimize':
        # Hardware Check
        if not _require_nvidia():
            log.error("Optimization requires an NVIDIA GPU and nvidia-smi.")
            sys.exit(1)
            
        pattern = '*' if args.model == 'all' else args.model
        inv = get_available_models()
        inv_names = [m.model for m in inv]
        inv_set = set(inv_names)
        matches = fnmatch.filter(inv_names, pattern)
        
        is_batch = '*' in pattern or '?' in pattern or len(matches) > 1
        if is_batch:
            print(f"Processing {len(matches)} models (Debug={'ON' if args.debug else 'OFF'})...")
            print("Legend: [SKIP] Ignored | [OK] Optimized | [ERR] Failed")
            print("-" * 60)

        for m in matches:
            if not is_batch: print(f"--- {m} ---")
            calculate_max_context(m, inv_set, args.buffer, is_batch, args.debug, args.manual)
    
    elif args.command == 'rm':
        delete_models_interactive(args.pattern, args.yes)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()