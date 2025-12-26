import os, subprocess, json, re, time
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ensure_dir, save_json
ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / 'external' / 'LWE-benchmarking'   # must clone here
RESULTS = ROOT / 'results' / 'salsa_runs'
ensure_dir(str(RESULTS))

# Update this path if the external repo layout differs
EXTERNAL_TRAIN_SCRIPT = EXTERNAL / 'src' / 'salsa' / 'train_and_recover.py' 
# fallback path based on README example
if not EXTERNAL_TRAIN_SCRIPT.exists():
    EXTERNAL_TRAIN_SCRIPT = EXTERNAL / 'src' / 'salsa' / 'train_and_recover.py'

if not EXTERNAL_TRAIN_SCRIPT.exists():
    print('ERROR: Could not find train_and_recover.py in external/LWE-benchmarking repo. Please clone the repo and check path.')
    print('Expected at:', EXTERNAL_TRAIN_SCRIPT)
    raise SystemExit(1)

# Load datasets from config file
config_path = Path(__file__).resolve().parent / 'configs' / 'light_params.json'
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    datasets = config.get('datasets', [])
    print(f"✅ Loaded {len(datasets)} datasets from {config_path.name}")
else:
    print(f"⚠️  Config file not found: {config_path}")
    print("   Using default datasets...")
    datasets = [
        {'name': 'n10', 'n': 10, 'q': 842779, 'm': 500, 'sigma': 3.0, 'hamming': 3, 'seed': 111},
        {'name': 'n30', 'n': 30, 'q': 842779, 'm': 2000, 'sigma': 3.0, 'hamming': 3, 'seed': 222}
    ]

def build_cmd(data_path, exp_name, seed):
    flags = {}
    # Use the venv python directly
    python_exe = str(ROOT / '.venv' / 'bin' / 'python3')
    dump_path = ROOT / 'results' / 'salsa_dumps'
    dump_path.mkdir(parents=True, exist_ok=True)
    # Build command to run from EXTERNAL directory so src is importable
    cmd = [
        'bash', '-c',
        f'cd {EXTERNAL} && {python_exe} src/salsa/train_and_recover.py ' +
        f'--data_path {data_path} ' +
        f'--exp_name {exp_name} ' +
        f'--secret_seed {seed} ' +
        f'--dump_path {dump_path} ' +
        f'--rlwe 0 ' +
        f'--task lwe ' +
        f'--angular_emb true ' +
        f'--dxdistinguisher true ' +
        f'--hamming 3 ' +
        f'--max_samples 300000 ' +
        f'--train_batch_size 128 ' +
        f'--val_batch_size 256 ' +
        f'--n_enc_heads 16 ' +
        f'--n_enc_layers 1 ' +
        f'--enc_emb_dim 1024 ' +
        f'--dec_emb_dim 512 ' +
        f'--n_dec_heads 4 ' +
        f'--use_ut true ' +
        f'--enc_loops 8 ' +
        f'--dec_loops 8 ' +
        f'--enc_gated true ' +
        f'--enc_act true ' +
        f'--enc_loop_idx 0 ' +
        f'--optimizer adam,lr=0.00005 ' +
        f'--epochs 50 ' +
        f'--check_secret_every 500 ' +
        f'--distinguisher_size 128 ' +
        f'--compile false'
    ]
    return cmd

# find precomputed dataset folders
pre_dir = ROOT / 'data' / 'precomputed'
folders = sorted([p for p in pre_dir.iterdir() if p.is_dir() and 'baseline' in p.name])  # baseline만 실행

# run SALSA on each folder containing baseline_*
for f in tqdm(folders, desc='salsa_folders'):
    # set experiment name
    exp_name = f.name + '_exp'
    out_dir = RESULTS / f.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(f, exp_name, seed=0)
    print(f"\n{'='*80}")
    print(f"🚀 {exp_name} 실행 중...")
    print(f"{'='*80}")
    
    # run and capture stdout/stderr with live epoch progress bar
    try:
        epochs_count = int(re.search(r'--epochs\s+(\d+)', ' '.join(cmd)).group(1))
    except Exception:
        epochs_count = 1
    epoch_bar = tqdm(total=epochs_count, desc=f'{f.name} epochs', leave=True)
    recovery_count = 0

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
    stdout_lines = []

    for line in p.stdout:
        stdout_lines.append(line)
        l = line.rstrip()

        # epoch start detection -> update epoch progress bar
        if 'starting epoch' in l.lower() or 'starting epoch' in line:
            # increment epoch counter
            try:
                epoch_bar.update(1)
            except Exception:
                pass
            print(f"  📊 {l}")
        elif 'loss' in l.lower() or 'train/' in l or '"train/' in l:
            # 학습률, 손실값, 에포크 정보 추출
            print(f"  📈 {l}")
        elif 'starting secret recovery' in l.lower() or 'secret recovery' in l.lower():
            recovery_count += 1
            tqdm.write(f"  ⏳ [secret recovery #{recovery_count}] {l}")
        elif 'Best secret guess' in line:
            print(f"  ✅ {l}")
        elif 'recovered' in l.lower():
            print(f"  🎯 {l}")

    try:
        epoch_bar.close()
    except Exception:
        pass
    
    ret = p.wait()
    
    print(f"\n{'='*80}")
    print(f"✅ {exp_name} 완료 (코드: {ret})")
    print(f"{'='*80}\n")
    
    save_json(out_dir / 'run_stdout.json', {'returncode': ret, 'stdout': stdout_lines})
    # try to parse predicted secret(s) from stdout (simple regex)
    joined = ''.join(stdout_lines)
    # look for "Best secret guess" patterns and extract numbers in brackets
    guesses = re.findall(r'Best secret guess[^\[]*\[([^\]]+)\]', joined)
    parsed = []
    for g in guesses:
        # split numbers, convert to ints
        parts = re.findall(r'-?\d+', g)
        parsed.append([int(x) for x in parts])
    if parsed:
        save_json(out_dir / 'predicted_secrets.json', {'guesses': parsed})
    else:
        # no parsed guesses; save raw stdout for manual inspection
        save_json(out_dir / 'parsed_guess_error.json', {'note': 'no guesses parsed', 'inspect_stdout': str(out_dir / 'run_stdout.json')})
    # also save a small metadata file
    save_json(out_dir / 'run_meta.json', {'folder': str(f), 'cmd': ' '.join(cmd), 'returncode': ret})

print('All SALSA runs completed. Results under results/salsa_runs/')
