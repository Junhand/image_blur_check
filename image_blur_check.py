import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# ------------------ ブレ指標 ------------------

def laplacian_variance_score(gray: np.ndarray) -> float:
    # 低いほどブレ（エッジが少ない）
    # Ref: Variance of Laplacian (OpenCV/PyImageSearch)
    lap = cv2.Laplacian(gray, ddepth=cv2.CV_64F, ksize=3)
    return float(lap.var())

def tenengrad_score(gray: np.ndarray, ksize: int = 3) -> float:
    # 低いほどブレ（勾配エネルギが小さい）
    # Ref: Tenengrad / Sobel Energy
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    g2 = gx * gx + gy * gy
    return float(np.mean(g2))


def fft_highfreq_ratio(gray: np.ndarray, hp_radius_ratio: float = 0.08, save_path: str = "fft_debug.png") -> float:
    # FFT
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag2 = np.abs(fshift) ** 2
    h, w = gray.shape[:2]
    cy, cx = h // 2, w // 2
    r = int(min(h, w) * hp_radius_ratio)

    Y, X = np.ogrid[:h, :w]
    mask_low = (Y - cy) ** 2 + (X - cx) ** 2 <= r * r

    total = mag2.sum()
    high = mag2[~mask_low].sum()
    
    # === 可視化部分（複素数→振幅の対数で表示）===
    eps = 1e-8
    viz_f = np.log1p(np.abs(f) + eps)
    viz_fshift = np.log1p(np.abs(fshift) + eps)
    viz_mag2 = np.log1p(mag2 + eps)

    plt.figure(figsize=(15, 5))

    # f: 周波数成分（対数振幅）
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(viz_f, cmap="Reds")
    ax1.set_title("FFT (log |f|)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # fshift: 中央シフト後（対数振幅）+ 低周波マスクの半径を可視化
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(viz_fshift, cmap="Reds")
    ax2.set_title("FFT Shifted (log |fshift|)")
    circ = Circle((cx, cy), r, fill=False, linewidth=1.5)
    ax2.add_patch(circ)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # mag2: パワースペクトル（対数）
    ax3 = plt.subplot(1, 3, 3)
    im3 = ax3.imshow(viz_mag2, cmap="Reds")
    ax3.set_title("Power Spectrum (log mag^2)")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()

    # 画像保存
    image_root = Path("output_image")
    image_root.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{image_root}/{save_path}", dpi=150)
    plt.close()

    print(f"FFT 可視化結果を {save_path} に保存しました。")

    return float(high / (total + 1e-8))

# ------------------ ユーティリティ ------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def list_images(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    else:
        return [p for p in root.iterdir() if p.suffix.lower() in IMG_EXTS]

def load_and_preprocess(path: Path, target_width: int = 300) -> np.ndarray:
    """
    画像を読み込み、カラーならグレースケール変換せずにそのまま、
    あとでグレースケール化必要なら行う。
    まずサイズを横幅 = target_width にリサイズ（縦はアスペクト比維持）。
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img

def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return img

def score_one(gray: np.ndarray, file_name: str) -> Tuple[float, float, float, float | None]:
    lv = laplacian_variance_score(gray)
    tg = tenengrad_score(gray)
    fr = fft_highfreq_ratio(gray, hp_radius_ratio=0.08, save_path=f"fft_debug_{file_name}.png")
    return lv, tg, fr

# ------------------ メイン ------------------

def main():
    ap = argparse.ArgumentParser(description="Batch blur scoring & removal with user-defined thresholds.")
    ap.add_argument("--input", type=str, required=True, help="input folder containing frames")
    ap.add_argument("--recursive", action="store_true", help="recurse into subfolders")
    ap.add_argument("--csv", type=str, default=None, help="save scores CSV path")
    ap.add_argument("--dry-run", action="store_true", help="only score & print; no file ops")
    ap.add_argument("--move-to", type=str, default=None, help="move blurry images to this subfolder under input (e.g., _BLUR)")
    ap.add_argument("--delete", action="store_true", help="delete blurry images (danger!)")
    ap.add_argument("--width", type=int, default=300, help="width to normalize to (pixels), height scaled to maintain aspect ratio")
    
    # --- 閾値（この値“以下”ならブレ） ---
    ap.add_argument("--thr-lap", type=float, default=5000.0, help="threshold for Laplacian variance (<= is blur)")
    ap.add_argument("--thr-ten", type=float, default=10000.0, help="threshold for Tenengrad Sobel energy (<= is blur)")
    ap.add_argument("--thr-fft", type=float, default=0.12, help="threshold for FFT high-frequency ratio (<= is blur)")

    # 多数決：ブレ票が threshold 以上でブレ
    ap.add_argument("--votes", type=int, default=2, choices=[1,2,3],
                    help="number of metrics that must vote 'blur' to mark as blur")
    args = ap.parse_args()

    root = Path(args.input)
    paths = list_images(root, args.recursive)
    if not paths:
        print("No images found.")
        return

    rows = []
    for p in paths:
        try:
            file_name = p.stem
            img_color = load_and_preprocess(p, target_width=args.width)
            gray = to_gray(img_color)
            lv, tg, fr = score_one(gray, file_name=file_name)
            rows.append((p, lv, tg, fr))
        except Exception as e:
            print(f"[WARN] {p}: {e}")

    # 閾値の表示
    print(f"[Thresholds]")
    print(f"  lap_var <= {args.thr_lap:.3f}  |  tenengrad <= {args.thr_ten:.3f}  |  fft_ratio <= {args.thr_fft:.4f}")
    print(f"[Rule] mark as BLUR if votes >= {args.votes}  (L/T/FFT: <= thr → ブレ,  BE: >= thr → ブレ)")

    # 仕分け
    blur_list, sharp_list = [], []
    for p, lv, tg, fr in rows:
        votes = int(lv <= args.thr_lap) + int(tg <= args.thr_ten) + int(fr <= args.thr_fft)
        is_blur = votes >= args.votes
        (blur_list if is_blur else sharp_list).append((p, lv, tg, fr, votes))

    print(f"[Summary] total={len(rows)} | blur={len(blur_list)} | sharp={len(sharp_list)}")

    # CSV 出力
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "lap_var", "tenengrad", "fft_ratio", "votes", "label"])
            for p, lv, tg, fr, v in blur_list:
                w.writerow([str(p), lv, tg, fr, v, "blur"])
            for p, lv, tg, fr, v in sharp_list:
                w.writerow([str(p), lv, tg, fr, v, "sharp"])
        print(f"[Saved] {csv_path}")

    # ファイル操作
    if args.dry_run:
        print("[Dry-run] No file operations performed.")
        return

    if args.delete and args.move_to:
        raise SystemExit("Choose either --delete or --move-to, not both.")

    if args.delete:
        removed = 0
        for p, *_ in blur_list:
            try:
                os.remove(p)
                removed += 1
            except Exception as e:
                print(f"[WARN] remove failed {p}: {e}")
        print(f"[Removed] {removed} blurred images")
    elif args.move_to:
        dest_root = root / args.move_to
        dest_root.mkdir(parents=True, exist_ok=True)
        moved = 0
        for p, *_ in blur_list:
            try:
                dest = dest_root / p.name
                if dest.exists():  # 重複名回避
                    stem, ext = dest.stem, dest.suffix
                    i = 1
                    while True:
                        cand = dest_root / f"{stem}_{i}{ext}"
                        if not cand.exists():
                            dest = cand; break
                        i += 1
                p.rename(dest)
                moved += 1
            except Exception as e:
                print(f"[WARN] move failed {p}: {e}")
        print(f"[Moved] {moved} blurred images -> {dest_root}")
    else:
        print("[Info] Neither --delete nor --move-to specified. Nothing moved/removed.")

if __name__ == "__main__":
    main()