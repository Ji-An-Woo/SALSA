# 데이터 생성
import numpy as np
from scipy.signal import fftconvolve
import os, json, csv
from tqdm import tqdm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'precomputed'
OUT.mkdir(parents=True, exist_ok=True)

def sample_secret(n, hamming=3, seed=None): # 비밀키의 길이, 비밀키의 1의 개수, 난수 생성 시드값
    rng = np.random.RandomState(seed) #난수생성기 none이면 매 시간 완전 랜덤
    s = np.zeros(n, dtype=np.int64) # s크기가 n인 0으로 채워진 배열 생성
    ones = rng.choice(n, size=hamming, replace=False) # 0부터 n-1까지 숫자 중에서 해밍 갯수 만큼 중복 없이 랜덤 선택, 같은 인덱스를 두번 선택하지 않음
    s[ones] = 1 # ones에 있는 인덱스 위치에 1을 설정 그러니까  s = [0, 0, 0, 0, 0], ones = [2, 7, 5], 실행 후 = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    return s # s는 1이 정확히 3개 있는 배열을 반환 // 무작위로 만들어진 비밀키

def circular_wrap(conv_result, n, q): # 원형으로 래핑
    a = conv_result[:n] #컨볼루전 결과의 처음 n개 원소만 추출함
    b = conv_result[n:2*n] if conv_result.shape[0] >= 2*n else np.zeros(n, dtype=np.int64) #컨볼루션 결과의 n번째 부터 2n번째 원소까지 추출, 만약 길이가 2n보다 작으면 0으로 채워진 길이 n짜리 배열 생성
    return (a + b) % q  # a와 b를 원소별로 더한 후 q로 나눈 나머지를 반환

def obfuscate_maclaurin(s, q, degrees=[1,3,5], coeffs=None, coeff_choices=[-1,1]): # 원본 비밀키, 모듈로 연산 10나누기 2가 2보다 커질 수 없으니까 이 원리, 
    n = len(s) # 비밀키의 길이를 n에 저장
    rng = np.random.RandomState(0) #reg 변수를 랜덤시드 0으로 고정
    if coeffs is None: # 계수가 주어지지 않을 경우
        coeffs = {d: int(rng.choice(coeff_choices)) for d in degrees} # -1 1 중에 랜덤 시드가 적용된걸로 적용
    s = s.astype(np.int64) # s를 int64 타입으로 변환
    s_prime = np.zeros(n, dtype=np.int64) # s'은 0으로 채워진 길이 n짜리 배열 생성, 누적 계산을 위함, 
    for d in degrees: # 각 차수에 대해
        if d == 1: # 1차항일 경우
            term = s.copy() # 원본 비밀키 복사
        else: # 2차항 이상일 경우
            term = s.copy() # 원본 비밀키 복사
            for _ in range(d-1): # 차수-1 만큼 반복
                conv = fftconvolve(term, s, mode='full').astype(np.int64) #푸리에 변환을 이용한 컨볼루션 연산
                term = circular_wrap(conv, n, 842779) # 원형 래핑을 통해 길이 n로 조정
        s_prime = (s_prime + coeffs[d] * term) % 842779 # 계수를 곱한 항을 누적하여 s' 계산
    return s_prime % 842779, coeffs # 최종 obfuscated 비밀키와 사용된 계수 반환

def gen_lwe_samples(n, q, m, sigma, s, seed=None): #lwe 샘플 생성
    rng = np.random.RandomState(seed) #난수생성기
    A = rng.randint(low=0, high=q, size=(m,n), dtype=np.int64) # m x n 크기의 균등분포 정수 행렬 생성 분포의 범위는 0부터 q-1까지
    e = np.round(rng.normal(loc=0.0, scale=sigma, size=(m,))).astype(np.int64) % q #정규 분포 내에서 랜덤 잡음 m개를 생성, 평균 0, 표준편차 sigma, 정수로 반올림 후 q로 나눈 나머지
    b = (A.dot(s) + e) % q #벡터 내적
    return A, b, e #행렬 A와 비밀키 s의 내적에 잡음 e를 더한 후 q로 나눈 나머지 반환

def save_npy(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, obj)

def write_csv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Get all unique keys across all rows
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    header = sorted(all_keys)
    import csv
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def main():
    cfg = json.load(open('configs/light_params.json')) #데이터셋 파라미터 불러오기
    rows = [] #생성된 데이터셋 정보를 저장할 리스트
    print('Generating precomputed datasets...') #데이터셋 생성 시작
    for ds in tqdm(cfg['datasets'], desc='datasets'): #각 데이터셋 파라미터에 대해 반복
        name = ds['name'] #데이터셋 이름
        n = ds['n']; q = ds['q']; m = ds['m']; sigma = ds['sigma']; hamming = ds['hamming']; seed = ds['seed'] #각 파라미터 추출
        
        # Generate data for baseline
        s = sample_secret(n, hamming=hamming, seed=seed) #비밀키 생성
        A_orig, b_orig, e_orig = gen_lwe_samples(n, q, m, sigma, s, seed=seed+1) #전체 LWE 샘플 생성
        
        # Split data: train (60%), val (20%), test (20%)
        train_size = int(0.6 * m)
        val_size = int(0.2 * m)
        
        A_train, b_train = A_orig[:train_size], b_orig[:train_size]
        A_val, b_val = A_orig[train_size:train_size+val_size], b_orig[train_size:train_size+val_size]
        A_test, b_test = A_orig[train_size+val_size:], b_orig[train_size+val_size:]
        
        outdir = OUT / f'baseline_{name}' #기본 데이터셋 저장 경로 설정
        outdir.mkdir(parents=True, exist_ok=True) #디렉토리 생성
        
        # Save SALSA-compatible format: *_A.npy in parent dir, *_b_{hamming}_{seed}.npy in data dir
        save_npy(A_train, OUT / 'train_A.npy')
        save_npy(A_val, OUT / 'val_A.npy')
        save_npy(A_test, OUT / 'test_A.npy')
        save_npy(A_orig, OUT / 'orig_A.npy')
        
        save_npy(b_train, outdir / f'train_b_{hamming}_{seed}.npy')
        save_npy(b_val, outdir / f'val_b_{hamming}_{seed}.npy')
        save_npy(b_test, outdir / f'test_b_{hamming}_{seed}.npy')
        save_npy(b_orig, outdir / f'orig_b_{hamming}_{seed}.npy')
        
        # Also save the original versions without split
        save_npy(A_orig, outdir / 'A.npy')
        save_npy(b_orig, outdir / 'b.npy')
        save_npy(e_orig, outdir / 'e.npy')
        
        json.dump({'s': s.tolist(), 'params': ds}, open(outdir / 'meta.json','w'), indent=2) #메타데이터 저장
        rows.append({'type':'baseline','name':name,'n':n,'m':m,'path':str(outdir)}) #생성된 데이터셋 정보 기록

        # Generate data for IDEA (obfuscated)
        s_prime, coeffs = obfuscate_maclaurin(s, q, degrees=cfg['idea_params']['degrees'], coeff_choices=cfg['idea_params']['coeff_choices']) #obfuscation 함수 참고
        A2_orig, b2_orig, e2_orig = gen_lwe_samples(n, q, m, sigma, s_prime, seed=seed+2) #obfuscated 비밀키로 LWE 샘플 생성
        
        A2_train, b2_train = A2_orig[:train_size], b2_orig[:train_size]
        A2_val, b2_val = A2_orig[train_size:train_size+val_size], b2_orig[train_size:train_size+val_size]
        A2_test, b2_test = A2_orig[train_size+val_size:], b2_orig[train_size+val_size:]
        
        outdir2 = OUT / f'idea_{name}' #obfuscated 데이터셋 저장 경로 설정
        outdir2.mkdir(parents=True, exist_ok=True) #디렉토리 생성
        
        # Note: We share the A matrices across baseline and idea
        # (reusing train_A.npy, etc. from above)
        save_npy(b2_train, outdir2 / f'train_b_{hamming}_{seed}.npy')
        save_npy(b2_val, outdir2 / f'val_b_{hamming}_{seed}.npy')
        save_npy(b2_test, outdir2 / f'test_b_{hamming}_{seed}.npy')
        save_npy(b2_orig, outdir2 / f'orig_b_{hamming}_{seed}.npy')
        
        save_npy(A2_orig, outdir2 / 'A.npy')
        save_npy(b2_orig, outdir2 / 'b.npy')
        save_npy(e2_orig, outdir2 / 'e.npy')
        
        json.dump({'s': s.tolist(), 's_prime': s_prime.tolist(), 'coeffs': coeffs, 'params': ds}, open(outdir2 / 'meta.json','w'), indent=2) #메타데이터 저장 
        rows.append({'type':'idea','name':name,'n':n,'m':m,'degrees':str(cfg['idea_params']['degrees']),'coeffs':str(coeffs),'path':str(outdir2)}) #생성된 데이터셋 정보 기록

    write_csv(rows, OUT / 'generated_datasets_params.csv')
    print('Saved precomputed datasets in', OUT)

if __name__ == '__main__':
    main()
