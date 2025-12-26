import json, os, csv, re
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from utils import ensure_dir
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results' / 'salsa_runs'
PLOTS = ROOT / 'results' / 'plots'
ensure_dir(str(PLOTS))

def load_json(p): # JSON 파일 불러오기 p는 파일 경로
    if Path(p).exists():
        return json.load(open(p))
    return None

def compute_recovery(true_s, pred_s): # 실제 비밀키와 예측된 비밀키를 비교하여 복구 정확도 계산
    true = np.array(true_s) # 실제 비밀키 배열화
    pred = np.array(pred_s) # 예측된 비밀키 배열화
    exact = int(np.array_equal(true, pred)) # 두 배열이 완전히 일치하는지 확인, 일치하면 1, 아니면 0
    bitwise = float((true == pred).sum()) / true.shape[0] # 각 비트별로 일치하는지 확인하여 일치하는 비트의 비율 계산
    return exact, bitwise # 정확한 복구 여부와 비트 단위 복구 비율 반환

if __name__ == '__main__':
    folders = sorted([p for p in RESULTS.iterdir() if p.is_dir()])
    summary = []
    for f in tqdm(folders, desc='eval_folders'): # 각 폴더에 대해 평가 수행, 메인로직 시작이라고 보면 됨
        meta = load_json(f / 'run_meta.json') or {} # 메타데이터 로드
        parsed = load_json(f / 'predicted_secrets.json') or {} # 예측된 비밀키 로드
        # load ground truth from precomputed data folder
        pre_meta_path = Path(meta.get('folder','')) / 'meta.json' # 사전 계산된 메타데이터 경로 설정
        true_s = None # 실제 비밀키 초기화
        s_prime = None # obfuscated 비밀키 초기화
        if pre_meta_path.exists(): 
            pre_meta = load_json(pre_meta_path)
            true_s = pre_meta.get('s', None) # 실제 비밀키 로드 아니면 None
            s_prime = pre_meta.get('s_prime', None) # obfuscated 비밀키 로드 아니면 None
        
        # evaluate parsed guesses if any
        exact_vs_s, bitwise_vs_s = None, None # 실제 비밀키에 대한 복구 메트릭 초기화
        exact_vs_s_prime, bitwise_vs_s_prime = None, None # obfuscated 비밀키에 대한 복구 메트릭 초기화
        
        if parsed and true_s is not None: # 예측된 비밀키가 있고 실제 비밀키가 있을 때
            best = parsed.get('guesses', parsed) if isinstance(parsed, dict) else parsed # 예측된 비밀키 추출
            # consider first guess
            guess = best[0] if isinstance(best, list) and len(best)>0 else None # 첫 번째 예측된 비밀키 선택
            if guess is not None: # 예측된 비밀키가 있을 때
                # Always measure vs original s (for all datasets)
                exact_vs_s, bitwise_vs_s = compute_recovery(true_s, guess) # 실제 비밀키와 예측된 비밀키 비교하여 복구 정확도 계산
                
                # For idea datasets (when s_prime exists), also measure vs s_prime
                if s_prime is not None: # obfuscated 비밀키가 있을 때
                    exact_vs_s_prime, bitwise_vs_s_prime = compute_recovery(s_prime, guess) # obfuscated 비밀키와 예측된 비밀키 비교하여 복구 정확도 계산
        
        # write summary row with both metrics
        summary.append({ # 요약 행 작성
            'folder': f.name,
            'exact_recovery_vs_s': exact_vs_s,
            'bitwise_recovery_vs_s': bitwise_vs_s,
            'exact_recovery_vs_s_prime': exact_vs_s_prime,
            'bitwise_recovery_vs_s_prime': bitwise_vs_s_prime,
            'meta': meta
        })
    # save CSV and JSON
    with open(RESULTS / 'salsa_summary.csv','w',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['folder','exact_recovery_vs_s','bitwise_recovery_vs_s','exact_recovery_vs_s_prime','bitwise_recovery_vs_s_prime','meta'])
        writer.writeheader()
        for r in summary:
            writer.writerow(r)
    json.dump(summary, open(RESULTS / 'salsa_summary.json','w'), indent=2)
    print('Saved salsa_summary.csv and salsa_summary.json; parsed guesses (if any) in each run folder.')
