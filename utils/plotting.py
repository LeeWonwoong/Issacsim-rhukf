"""
plotting.py — 학습 결과 시각화
================================
오프라인 플로팅 유틸리티. 학습 곡선, 에피소드 리플레이 등.
"""
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_training_progress(rewards, losses, outdir='./results'):
    if not HAS_MPL:
        print("[plotting] matplotlib not available, skipping.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(rewards, 'b-', alpha=0.3, lw=1)
    if len(rewards) >= 20:
        ma = np.convolve(rewards, np.ones(20) / 20, 'valid')
        ax.plot(range(19, len(rewards)), ma, 'b-', lw=2)
    ax.set_title('Episode Reward'); ax.set_xlabel('Episode')
    ax = axes[1]
    ax.plot(losses, 'r-', alpha=0.7, lw=1)
    ax.set_title('TD Loss'); ax.set_xlabel('Episode')
    plt.tight_layout()
    path = f"{outdir}/training_progress.png"
    plt.savefig(path, dpi=100); plt.close()
    print(f"[plotting] Saved: {path}")
