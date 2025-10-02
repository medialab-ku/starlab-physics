import numpy as np
import os

# 상대경로를 절대경로로 변환
dir = os.path.abspath("./pillar-dt0.00200-tol4-opt1000-cfl-warmstart-errl2/")

bender_stats = {
    "elapsed_time"      : "pillar-dt0.00200-tol4-opt1000-cfl-2014Bender-elapsed_time_ms.npy",
    "opt_iter"          : "pillar-dt0.00200-tol4-opt1000-cfl-2014Bender-opt_iter.npy",
    "pcg_iter"          : "pillar-dt0.00200-tol4-opt1000-cfl-2014Bender-pcg_iter.npy",
}

ours_stats = {
    "elapsed_time"      : "pillar-dt0.00200-tol4-opt1000-cfl-errl2-warmstart-ours-elapsed_time_ms.npy",
    "opt_iter"          : "pillar-dt0.00200-tol4-opt1000-cfl-errl2-warmstart-ours-opt_iter.npy",
    "pcg_iter"          : "pillar-dt0.00200-tol4-opt1000-cfl-errl2-warmstart-ours-pcg_iter.npy",
}

for key in bender_stats.keys():
    bender_file = os.path.join(dir, bender_stats[key])
    ours_file = os.path.join(dir, ours_stats[key])

    # 파일 로드 및 평균 계산
    bender_data = np.load(bender_file)
    ours_data = np.load(ours_file)
    bender_mean = np.mean(bender_data)
    ours_mean = np.mean(ours_data)

    # 출력
    print(f"{key}:")
    print(f"  Bender's Average = {bender_mean:.2f}")
    print(f"  Ours' Average = {ours_mean:.2f}")
    print()