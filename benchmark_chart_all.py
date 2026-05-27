"""
benchmark_chart_all.py — Gộp tất cả đồ thị benchmark vào 1 file
Output tổ chức theo folder:
  charts/
    01_fps_timeline/       — FPS theo thời gian (từng method + tổng hợp)
    02_avg_fps/            — Bar chart FPS trung bình
    03_timing_breakdown/   — Stacked bar thời gian từng bước
    04_kde_distribution/   — Phân bố Gaussian/KDE
    05_cdf_cumulative/     — Đồ thị tích lũy
    06_box_plot/           — Box plot latency
    07_heatmap/            — Heatmap inference theo segment
    08_percentile/         — P50/P95/P99 bar chart
    09_summary_table/      — Bảng tóm tắt
    10_dashboard/          — Dashboard tổng hợp

Usage: uv run benchmark_chart_all.py
"""
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import stats

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['figure.dpi'] = 150

# CONFIG
base = os.getcwd()
LOG_DIR = os.path.join(base, "benchmark_logs")
CHARTS_ROOT = os.path.join(LOG_DIR, "charts")
WARMUP_FRAMES = 10

METHOD_COLORS = {
    'gstreamer_single': '#FF6B6B',
    'gstreamer_multi':  '#4ECDC4',
    'opencv':           '#45B7D1',
}
METHOD_LABELS = {
    'gstreamer_single': 'GStreamer (Single-Thread)',
    'gstreamer_multi':  'GStreamer (Multi-Thread)',
    'opencv':           'OpenCV (Pure)',
}

# Tạo subfolder
FOLDERS = {
    'fps_timeline':     '01_fps_timeline',
    'avg_fps':          '02_avg_fps',
    'timing_breakdown': '03_timing_breakdown',
    'kde':              '04_kde_distribution',
    'cdf':              '05_cdf_cumulative',
    'box_plot':         '06_box_plot',
    'heatmap':          '07_heatmap',
    'percentile':       '08_percentile',
    'summary':          '09_summary_table',
    'dashboard':        '10_dashboard',
}

def get_dir(key):
    d = os.path.join(CHARTS_ROOT, FOLDERS[key])
    os.makedirs(d, exist_ok=True)
    return d

def save(fig, folder_key, filename):
    path = os.path.join(get_dir(folder_key), filename)
    fig.savefig(path, bbox_inches='tight')
    print(f"  [SAVED] {path}")
    plt.close(fig)

# LOAD DATA
csv_files = {
    'gstreamer_single': os.path.join(LOG_DIR, 'benchmark_log_gstreamer_single.csv'),
    'gstreamer_multi':  os.path.join(LOG_DIR, 'benchmark_log_gstreamer_multi.csv'),
    'opencv':           os.path.join(LOG_DIR, 'benchmark_log_opencv.csv'),
}

all_data = {}
for method, path in csv_files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        if len(df) > 0:
            all_data[method] = df
            print(f"[OK] {method}: {len(df)} frames")

if not all_data:
    print("[ERROR] Không tìm thấy CSV!")
    sys.exit(1)

video_fps = 25.0
for df in all_data.values():
    if 'video_source_fps' in df.columns:
        video_fps = df['video_source_fps'].iloc[0]
        break

def fw(df):
    return df[df['frame_id'] > WARMUP_FRAMES].copy()

# ============================================================
# 1. FPS TIMELINE — từng method riêng + tổng hợp
# ============================================================
def chart_fps_timeline():
    print("\n[1/10] FPS Timeline...")
    fps_types = [('ai_fps', 'AI FPS'), ('pipeline_fps', 'Pipeline FPS'), ('display_fps', 'Display FPS')]

    # Từng method riêng
    for method, df in all_data.items():
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        fig.suptitle(f'FPS Over Time — {METHOD_LABELS[method]}', fontsize=14, fontweight='bold')
        dc = fw(df)
        for ax, (col, title) in zip(axes, fps_types):
            smooth = dc[col].astype(float).rolling(20, min_periods=1).mean()
            ax.plot(dc['frame_id'], smooth, color=METHOD_COLORS[method], linewidth=1.5)
            ax.axhline(y=video_fps, color='#FFD93D', linestyle='--', linewidth=2, alpha=0.7,
                       label=f'Source ({video_fps:.0f} FPS)')
            ax.set_ylabel('FPS'); ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
        axes[-1].set_xlabel('Frame ID')
        plt.tight_layout()
        save(fig, 'fps_timeline', f'fps_{method}.png')

    # Tổng hợp
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('FPS Over Time — So sánh tất cả', fontsize=14, fontweight='bold')
    for ax, (col, title) in zip(axes, fps_types):
        for method, df in all_data.items():
            dc = fw(df)
            smooth = dc[col].astype(float).rolling(20, min_periods=1).mean()
            ax.plot(dc['frame_id'], smooth, color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method], linewidth=1.5, alpha=0.85)
        ax.axhline(y=video_fps, color='#FFD93D', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Source ({video_fps:.0f} FPS)')
        ax.set_ylabel('FPS'); ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    axes[-1].set_xlabel('Frame ID')
    plt.tight_layout()
    save(fig, 'fps_timeline', 'fps_all_combined.png')

# ============================================================
# 2. AVG FPS BAR
# ============================================================
def chart_avg_fps():
    print("[2/10] Avg FPS...")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Average FPS Comparison', fontsize=14, fontweight='bold')
    fps_cols = ['ai_fps', 'pipeline_fps', 'display_fps']
    fps_labels = ['AI FPS', 'Pipeline FPS', 'Display FPS']
    x = np.arange(len(fps_cols)); width = 0.2
    offsets = np.linspace(-width, width, len(all_data))
    for idx, (method, df) in enumerate(all_data.items()):
        dc = fw(df)
        means = [dc[c].astype(float).mean() for c in fps_cols]
        stds = [dc[c].astype(float).std() for c in fps_cols]
        bars = ax.bar(x + offsets[idx], means, width*0.9, yerr=stds, capsize=3,
                      color=METHOD_COLORS[method], label=METHOD_LABELS[method], alpha=0.85)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{m:.1f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.axhline(y=video_fps, color='#FFD93D', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Source ({video_fps:.0f} FPS)')
    ax.set_xticks(x); ax.set_xticklabels(fps_labels)
    ax.set_ylabel('FPS'); ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    save(fig, 'avg_fps', 'avg_fps_comparison.png')

# ============================================================
# 3. TIMING BREAKDOWN
# ============================================================
def chart_timing():
    print("[3/10] Timing Breakdown...")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Average Timing Breakdown (ms)', fontsize=14, fontweight='bold')
    methods_list = list(all_data.keys()); x = np.arange(len(methods_list)); width = 0.5
    timing_cols = ['preprocess_ms', 'inference_ms', 'postprocess_ms']
    timing_labels = ['Preprocess', 'Inference', 'Postprocess']
    timing_colors = ['#45B7D1', '#FF6B6B', '#4ECDC4']
    # Check decode
    has_decode = any(fw(df)['decode_ms'].astype(float).mean() > 0.01 for df in all_data.values())
    if has_decode:
        timing_cols.insert(0, 'decode_ms'); timing_labels.insert(0, 'Decode'); timing_colors.insert(0, '#FFD93D')
    bottom = np.zeros(len(methods_list))
    for col, lbl, clr in zip(timing_cols, timing_labels, timing_colors):
        vals = np.array([fw(all_data[m])[col].astype(float).mean() for m in methods_list])
        ax.bar(x, vals, width, bottom=bottom, label=lbl, color=clr, alpha=0.85)
        for i, v in enumerate(vals):
            if v > 0.5:
                ax.text(x[i], bottom[i]+v/2, f'{v:.1f}ms', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        bottom += vals
    for i in range(len(methods_list)):
        ax.text(x[i], bottom[i]+0.5, f'Total: {bottom[i]:.1f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([METHOD_LABELS[m] for m in methods_list], fontsize=9)
    ax.set_ylabel('Time (ms)'); ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    save(fig, 'timing_breakdown', 'timing_breakdown.png')

# ============================================================
# 4. KDE DISTRIBUTION — từng method + tổng hợp
# ============================================================
def chart_kde():
    print("[4/10] KDE Distribution...")
    # Từng method
    for method, df in all_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        dc = fw(df); vals = dc['inference_ms'].astype(float).values
        p99 = np.percentile(vals, 99); vc = vals[vals <= p99]
        ax.hist(vc, bins=60, density=True, color=METHOD_COLORS[method], alpha=0.3, edgecolor='white')
        kde = stats.gaussian_kde(vc)
        xr = np.linspace(vc.min(), vc.max(), 300)
        ax.plot(xr, kde(xr), color=METHOD_COLORS[method], linewidth=2.5)
        ax.axvline(np.median(vc), color='red', linestyle='--', alpha=0.7, label=f'Median: {np.median(vc):.1f}ms')
        ax.axvline(np.mean(vc), color='blue', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(vc):.1f}ms')
        ax.set_title(f'Inference Distribution — {METHOD_LABELS[method]}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Inference (ms)'); ax.set_ylabel('Density'); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save(fig, 'kde', f'kde_{method}.png')

    # Tổng hợp
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Inference Distribution (KDE) — So sánh', fontsize=14, fontweight='bold')
    for method, df in all_data.items():
        dc = fw(df); vals = dc['inference_ms'].astype(float).values
        p99 = np.percentile(vals, 99); vc = vals[vals <= p99]
        kde = stats.gaussian_kde(vc)
        xr = np.linspace(vc.min(), vc.max(), 300)
        ax.plot(xr, kde(xr), color=METHOD_COLORS[method], label=METHOD_LABELS[method], linewidth=2)
        ax.fill_between(xr, kde(xr), color=METHOD_COLORS[method], alpha=0.12)
    ax.set_xlabel('Inference (ms)'); ax.set_ylabel('Density'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, 'kde', 'kde_all_combined.png')

# ============================================================
# 5. CDF CUMULATIVE — từng method + tổng hợp
# ============================================================
def chart_cdf():
    print("[5/10] CDF Cumulative...")
    # Từng method
    for method, df in all_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        dc = fw(df); vals = np.sort(dc['inference_ms'].astype(float).values)
        cdf = np.arange(1, len(vals)+1) / len(vals) * 100
        ax.plot(vals, cdf, color=METHOD_COLORS[method], linewidth=2.5)
        for pct in [50, 95, 99]:
            pval = np.percentile(vals, pct)
            ax.axhline(y=pct, color='gray', linestyle='--', alpha=0.4)
            ax.axvline(x=pval, color='gray', linestyle=':', alpha=0.4)
            ax.plot(pval, pct, 'ro', markersize=6)
            ax.annotate(f'P{pct}: {pval:.1f}ms', (pval, pct), textcoords="offset points",
                       xytext=(10, -5), fontsize=9, fontweight='bold')
        ax.set_title(f'CDF — {METHOD_LABELS[method]}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Inference (ms)'); ax.set_ylabel('Cumulative %'); ax.grid(True, alpha=0.3); ax.set_ylim(0, 105)
        plt.tight_layout()
        save(fig, 'cdf', f'cdf_{method}.png')

    # Tổng hợp
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('CDF — Cumulative Distribution (So sánh)', fontsize=14, fontweight='bold')
    for method, df in all_data.items():
        dc = fw(df); vals = np.sort(dc['inference_ms'].astype(float).values)
        cdf = np.arange(1, len(vals)+1) / len(vals) * 100
        ax.plot(vals, cdf, color=METHOD_COLORS[method], label=METHOD_LABELS[method], linewidth=2)
    for pct in [50, 95, 99]:
        ax.axhline(y=pct, color='gray', linestyle='--', alpha=0.4)
        ax.text(ax.get_xlim()[0]+0.3, pct+1, f'P{pct}', fontsize=8, color='gray')
    ax.set_xlabel('Inference (ms)'); ax.set_ylabel('Cumulative %')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 105)
    plt.tight_layout()
    save(fig, 'cdf', 'cdf_all_combined.png')

# ============================================================
# 6. BOX PLOT — từng method + tổng hợp
# ============================================================
def chart_box():
    print("[6/10] Box Plot...")
    # Tổng hợp
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Inference Latency Distribution', fontsize=14, fontweight='bold')
    data_box, labels_box, colors_box = [], [], []
    for method, df in all_data.items():
        data_box.append(fw(df)['inference_ms'].astype(float).values)
        labels_box.append(METHOD_LABELS[method]); colors_box.append(METHOD_COLORS[method])
    bp = ax.boxplot(data_box, tick_labels=labels_box, patch_artist=True,
                    flierprops={'marker': '.', 'markersize': 2, 'alpha': 0.3})
    for patch, c in zip(bp['boxes'], colors_box): patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel('Inference (ms)'); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    save(fig, 'box_plot', 'box_plot_all.png')

# ============================================================
# 7. HEATMAP
# ============================================================
def chart_heatmap():
    print("[7/10] Heatmap...")
    seg = 100; methods = list(all_data.keys())
    max_s = max(len(fw(df)) // seg for df in all_data.values())
    if max_s == 0: return
    matrix = np.full((len(methods), max_s), np.nan)
    for i, m in enumerate(methods):
        vals = fw(all_data[m])['inference_ms'].astype(float).values
        for s in range(len(vals) // seg):
            matrix[i, s] = np.mean(vals[s*seg:(s+1)*seg])
    fig, ax = plt.subplots(figsize=(16, 3))
    fig.suptitle(f'Inference Heatmap (mỗi ô = {seg} frames)', fontsize=14, fontweight='bold')
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_xlabel(f'Segment (x{seg} frames)')
    plt.colorbar(im, ax=ax, shrink=0.8).set_label('Avg Inference (ms)')
    plt.tight_layout()
    save(fig, 'heatmap', 'inference_heatmap.png')

# ============================================================
# 8. PERCENTILE BARS
# ============================================================
def chart_percentile():
    print("[8/10] Percentile Bars...")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Inference Latency Percentiles', fontsize=14, fontweight='bold')
    pcts = [50, 95, 99]; x = np.arange(len(pcts)); width = 0.2
    methods = list(all_data.keys())
    offsets = np.linspace(-width, width, len(methods))
    for idx, m in enumerate(methods):
        vals = fw(all_data[m])['inference_ms'].astype(float)
        pvals = [vals.quantile(p/100) for p in pcts]
        bars = ax.bar(x+offsets[idx], pvals, width*0.9, color=METHOD_COLORS[m],
                      label=METHOD_LABELS[m], alpha=0.85)
        for bar, v in zip(bars, pvals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{v:.1f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([f'P{p}' for p in pcts])
    ax.set_ylabel('Inference (ms)'); ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    save(fig, 'percentile', 'percentile_bars.png')

# ============================================================
# 9. SUMMARY TABLE
# ============================================================
def chart_summary():
    print("[9/10] Summary Table...")
    fig, ax = plt.subplots(figsize=(14, 4)); ax.axis('off')
    fig.suptitle('Benchmark Summary', fontsize=14, fontweight='bold')
    cols = ['Method', 'Frames', 'Avg AI FPS', 'Avg Pipeline FPS', 'Avg Display FPS',
            'Avg Inf (ms)', 'P50', 'P95', 'P99']
    rows = []
    for m, df in all_data.items():
        dc = fw(df); inf = dc['inference_ms'].astype(float)
        rows.append([METHOD_LABELS[m], len(dc), f"{dc['ai_fps'].astype(float).mean():.1f}",
                      f"{dc['pipeline_fps'].astype(float).mean():.1f}",
                      f"{dc['display_fps'].astype(float).mean():.1f}",
                      f"{inf.mean():.1f}", f"{inf.quantile(0.5):.1f}",
                      f"{inf.quantile(0.95):.1f}", f"{inf.quantile(0.99):.1f}"])
    if rows:
        tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
        for j in range(len(cols)):
            tbl[0, j].set_facecolor('#2C3E50'); tbl[0, j].set_text_props(color='white', fontweight='bold')
    plt.tight_layout()
    save(fig, 'summary', 'summary_table.png')

# ============================================================
# 10. DASHBOARD — Tổng hợp 4 chart chính
# ============================================================
def chart_dashboard():
    print("[10/10] Dashboard...")
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('BENCHMARK DASHBOARD', fontsize=16, fontweight='bold', y=0.98)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    for m, df in all_data.items():
        dc = fw(df); vals = dc['inference_ms'].astype(float).values
        p99 = np.percentile(vals, 99); vc = vals[vals <= p99]
        kde = stats.gaussian_kde(vc); xr = np.linspace(vc.min(), vc.max(), 200)
        ax1.plot(xr, kde(xr), color=METHOD_COLORS[m], label=METHOD_LABELS[m], linewidth=2)
        ax1.fill_between(xr, kde(xr), color=METHOD_COLORS[m], alpha=0.1)
        sv = np.sort(vals); cdf = np.arange(1, len(sv)+1)/len(sv)*100
        ax2.plot(sv, cdf, color=METHOD_COLORS[m], label=METHOD_LABELS[m], linewidth=2)
        sm = dc['display_fps'].astype(float).rolling(30, min_periods=1).mean()
        ax3.plot(dc['frame_id'], sm, color=METHOD_COLORS[m], label=METHOD_LABELS[m], linewidth=1.2, alpha=0.8)

    ax1.set_title('KDE Distribution'); ax1.set_xlabel('Inference (ms)'); ax1.set_ylabel('Density')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    for p in [50,95,99]: ax2.axhline(y=p, color='gray', linestyle='--', alpha=0.4)
    ax2.set_title('CDF Cumulative'); ax2.set_xlabel('Inference (ms)'); ax2.set_ylabel('Cumulative %')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 105)
    ax3.axhline(y=video_fps, color='#FFD93D', linestyle='--', linewidth=2, alpha=0.7, label=f'Source ({video_fps:.0f})')
    ax3.set_title('Display FPS'); ax3.set_xlabel('Frame ID'); ax3.set_ylabel('FPS')
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3); ax3.set_ylim(bottom=0)
    # Box plot
    db, lb, cb = [], [], []
    for m, df in all_data.items():
        db.append(fw(df)['inference_ms'].astype(float).values)
        lb.append(METHOD_LABELS[m]); cb.append(METHOD_COLORS[m])
    bp = ax4.boxplot(db, tick_labels=lb, patch_artist=True, flierprops={'marker':'.','markersize':1,'alpha':0.2})
    for p, c in zip(bp['boxes'], cb): p.set_facecolor(c); p.set_alpha(0.6)
    ax4.set_title('Box Plot'); ax4.set_ylabel('Inference (ms)'); ax4.grid(True, axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, 'dashboard', 'dashboard.png')

# ============================================================
# EXPORT EXCEL
# ============================================================
def export_excel():
    print("\n[EXCEL] Exporting...")
    path = os.path.join(LOG_DIR, 'benchmark_results.xlsx')
    with pd.ExcelWriter(path, engine='openpyxl') as w:
        for m, df in all_data.items():
            df.to_excel(w, sheet_name=m[:31], index=False)
        rows = []
        for m, df in all_data.items():
            dc = fw(df); inf = dc['inference_ms'].astype(float)
            rows.append({'Method': METHOD_LABELS[m], 'Frames': len(dc),
                         'Avg AI FPS': dc['ai_fps'].astype(float).mean(),
                         'Avg Pipeline FPS': dc['pipeline_fps'].astype(float).mean(),
                         'Avg Display FPS': dc['display_fps'].astype(float).mean(),
                         'Avg Inference (ms)': inf.mean(), 'P50': inf.quantile(0.5),
                         'P95': inf.quantile(0.95), 'P99': inf.quantile(0.99)})
        pd.DataFrame(rows).to_excel(w, sheet_name='Summary', index=False)
    print(f"  [SAVED] {path}")

# MAIN
if __name__ == "__main__":
    print("=" * 60)
    print("  BENCHMARK CHART GENERATOR (ALL-IN-ONE)")
    print("=" * 60)

    chart_fps_timeline()
    chart_avg_fps()
    chart_timing()
    chart_kde()
    chart_cdf()
    chart_box()
    chart_heatmap()
    chart_percentile()
    chart_summary()
    chart_dashboard()
    export_excel()

    print(f"\n{'=' * 60}")
    print(f"  Output: {CHARTS_ROOT}")
    print(f"{'=' * 60}")
