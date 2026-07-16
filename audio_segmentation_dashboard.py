"""
Audio Segmentation Comparison Dashboard
========================================
Compares SOTA structural segmentation algorithms across multiple audio files
with interactive waveform playback and segment visualisation.

Run:
    pip install -r requirements.txt
    python audio_segmentation_dashboard.py

Then open http://localhost:7860 in your browser.

Optional (for extra algorithms):
    pip install msaf          # MSAF suite (Foote, SCLUSTER, OLDA, CNMF)
    pip install allin1 torch  # Transformer-based SOTA (Kim et al. 2023)
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import scipy.ndimage
import scipy.signal
import scipy.linalg
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
AUDIO_FILES = {
    "ticket":    "https://dorienherremans.com/drop/misc/ticket.mp3",
    "zen":       "https://dorienherremans.com/drop/misc/zen.mp3",
    "miikesnow": "https://dorienherremans.com/drop/misc/miikesnow.mp3",
    "ladytron":  "https://dorienherremans.com/drop/misc/ladytron.mp3",
    "fastcar":   "https://dorienherremans.com/drop/misc/fastcar.mp3",
    "help":      "https://dorienherremans.com/drop/misc/help.mp3",
}

AUDIO_DIR = Path("./audio")
AUDIO_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path("./processed_cache")
CACHE_DIR.mkdir(exist_ok=True)

HOP_LENGTH = 512
SR_TARGET = 22050  # resample to this for consistency

ALGO_COLORS = [
    "#E74C3C",  # Onset       – red
    "#2ECC71",  # Foote       – green
    "#3498DB",  # Laplacian   – blue
    "#F39C12",  # Spectral    – orange
    "#9B59B6",  # MSAF-Foote  – purple
    "#1ABC9C",  # MSAF-SCLUST – teal
    "#E67E22",  # MSAF-OLDA   – dark orange
    "#E91E63",  # Allin1      – pink
]


# ─────────────────────────────────────────────
# STEP 1 – DOWNLOAD AUDIO
# ─────────────────────────────────────────────
def download_audio_files() -> None:
    for name, url in AUDIO_FILES.items():
        path = AUDIO_DIR / f"{name}.mp3"
        if not path.exists():
            print(f"  Downloading {name}…", end=" ", flush=True)
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                path.write_bytes(r.content)
                print("✓")
            except Exception as e:
                print(f"✗ ({e})")


# ─────────────────────────────────────────────
# STEP 2 – ALGORITHM IMPLEMENTATIONS
# ─────────────────────────────────────────────

def _ensure_boundaries(boundaries: np.ndarray, duration: float) -> np.ndarray:
    """Guarantee 0.0 start and duration end, no duplicates."""
    b = np.concatenate([[0.0], boundaries, [duration]])
    b = np.unique(np.round(b, 4))
    return b[(b >= 0) & (b <= duration + 0.001)]


def _checkerboard_kernel(L: int) -> np.ndarray:
    """
    Gaussian-tapered checkerboard kernel of size (L × L) (Foote 2000).
    Quadrants: +1 top-left / bottom-right, -1 top-right / bottom-left.
    """
    half = L // 2
    g = scipy.signal.windows.gaussian(L, std=L / 8)
    G = np.outer(g, g)                 # Gaussian taper, shape (L, L)
    checker = np.ones((L, L))
    checker[:half, half:] = -1         # top-right
    checker[half:, :half] = -1         # bottom-left
    return G * checker                 # shape (L, L)


# ── Algorithm 1: Onset-based (baseline) ──────────────────────────────────────
def segment_onset(
    y: np.ndarray,
    sr: int,
    min_seg_duration: float = 4.0,
    **_,
) -> np.ndarray:
    """
    Librosa onset detection baseline.
    Keeps only onsets separated by at least min_seg_duration seconds.
    """
    onset_times = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=HOP_LENGTH, units="time"
    )
    duration = librosa.get_duration(y=y, sr=sr)
    filtered = [0.0]
    for t in onset_times:
        if t - filtered[-1] >= min_seg_duration:
            filtered.append(float(t))
    return _ensure_boundaries(np.array(filtered[1:]), duration)


# ── Algorithm 2: Foote Novelty ────────────────────────────────────────────────
def segment_foote(
    y: np.ndarray,
    sr: int,
    kernel_size: int = 64,
    min_seg_duration: float = 3.0,
    **_,
) -> np.ndarray:
    """
    Novelty-based segmentation via Gaussian-tapered checkerboard kernel
    applied to the chroma self-similarity matrix (Foote 2000).
    """
    duration = librosa.get_duration(y=y, sr=sr)

    # Feature: chroma CQT, L2-normalised per frame
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    norms = np.linalg.norm(C, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    C = C / norms

    # Self-similarity matrix (cosine)
    S = C.T @ C
    S = (S - S.min()) / (S.max() - S.min() + 1e-8)

    # Checkerboard convolution along diagonal
    # K is (L × L); slide it along the diagonal, centred on each frame t
    K    = _checkerboard_kernel(kernel_size)   # (L, L)
    L    = kernel_size
    half = L // 2
    N    = S.shape[0]
    novelty = np.zeros(N)
    for t in range(half, N - half):
        block = S[t - half : t + half, t - half : t + half]
        if block.shape == (L, L):
            novelty[t] = np.sum(block * K)

    # Smooth and peak-pick
    novelty = scipy.ndimage.gaussian_filter1d(novelty, sigma=5)
    min_dist = max(1, int(min_seg_duration * sr / HOP_LENGTH))
    peaks, _ = scipy.signal.find_peaks(
        novelty,
        distance=min_dist,
        height=np.percentile(novelty[novelty > 0], 55),
    )
    boundary_times = librosa.frames_to_time(peaks, sr=sr, hop_length=HOP_LENGTH)
    return _ensure_boundaries(boundary_times, duration)


# ── Algorithm 3: Laplacian Segmentation ──────────────────────────────────────
def segment_laplacian(
    y: np.ndarray,
    sr: int,
    k: int = 6,
    bandwidth: float = 0.5,
    **_,
) -> np.ndarray:
    """
    Beat-synchronous Laplacian segmentation (McFee & Ellis, ISMIR 2014).
    Combines chroma + MFCC features on a beat-synced recurrence matrix,
    applies path enhancement, then clusters via k-means on eigenvectors.
    """
    duration = librosa.get_duration(y=y, sr=sr)

    # Beat tracking
    _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    if len(beats) < 4:
        return _ensure_boundaries(np.array([]), duration)
    k = min(k, max(2, len(beats) // 3))

    # Beat-synced features
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP_LENGTH, n_mfcc=13)
    bc = librosa.util.sync(chroma, beats, aggregate=np.median)
    bm = librosa.util.sync(mfcc,   beats, aggregate=np.mean)
    features = normalize(np.vstack([bc, bm]).T)   # (n_beats, n_features)

    # Recurrence matrix with path enhancement
    R = librosa.segment.recurrence_matrix(
        features.T, mode="affinity", sym=True, bandwidth=bandwidth
    )
    # Median filter along lag axis then path enhancement
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    R  = df(R, size=(1, 7))
    R  = librosa.segment.path_enhance(R, 15, window="hann", n_filters=7)

    # Normalised graph Laplacian
    deg = R.sum(axis=1)
    deg[deg == 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    Ln = np.eye(len(deg)) - D_inv_sqrt @ R @ D_inv_sqrt

    # Eigenvectors (smallest k+1, skip constant)
    try:
        _, vecs = scipy.linalg.eigh(Ln, subset_by_index=[0, k])
    except Exception:
        evals, vecs = np.linalg.eigh(Ln)
        vecs = vecs[:, :k + 1]

    embedding = normalize(vecs[:, 1:])  # drop constant eigenvector

    # k-means clustering
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    seg_ids = km.fit_predict(embedding)

    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LENGTH)
    boundaries = [beat_times[i] for i in range(1, len(seg_ids))
                  if seg_ids[i] != seg_ids[i - 1] and i < len(beat_times)]
    return _ensure_boundaries(np.array(boundaries), duration)


# ── Algorithm 4: Spectral Clustering on SSM ──────────────────────────────────
def segment_spectral(
    y: np.ndarray,
    sr: int,
    n_segments: int = 8,
    **_,
) -> np.ndarray:
    """
    Spectral clustering on a downsampled frame-level self-similarity matrix
    built from combined chroma + MFCC features.
    """
    duration = librosa.get_duration(y=y, sr=sr)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP_LENGTH, n_mfcc=13)
    features = normalize(np.vstack([chroma, mfcc]).T)  # (n_frames, F)

    # Downsample to ≤300 frames for speed and stable clustering
    step = max(1, len(features) // 300)
    feat_ds = features[::step]

    S = feat_ds @ feat_ds.T
    S = np.clip(S, 0, 1)
    np.fill_diagonal(S, 0)

    n_seg = min(n_segments, max(2, len(feat_ds) // 8))
    sc = SpectralClustering(
        n_clusters=n_seg, affinity="precomputed", n_init=5, random_state=42
    )
    labels = sc.fit_predict(S + 1e-8)

    # Smooth label sequence to suppress single-frame noise before finding boundaries
    from scipy.stats import mode as scipy_mode
    smoothed = np.array([
        int(scipy_mode(labels[max(0, i-3) : i+4], keepdims=False).mode)
        for i in range(len(labels))
    ])

    boundaries = []
    for i in range(1, len(smoothed)):
        if smoothed[i] != smoothed[i - 1]:
            frame = i * step
            t = frame * HOP_LENGTH / sr
            if 0.0 < t < duration:
                boundaries.append(t)
    return _ensure_boundaries(np.array(boundaries), duration)


# ── Algorithm 5: Foote Novelty on Tempogram ──────────────────────────────────
def segment_foote_tempogram(
    y: np.ndarray,
    sr: int,
    kernel_size: int = 64,
    min_seg_duration: float = 3.0,
    **_,
) -> np.ndarray:
    """
    Foote Novelty applied to a tempogram SSM instead of chroma.
    Detects *rhythmic* structure breaks (tempo/rhythm changes) rather than
    harmonic ones — complementary to the chroma-based Foote variant.
    """
    duration = librosa.get_duration(y=y, sr=sr)

    tempogram = librosa.feature.tempogram(y=y, sr=sr, hop_length=HOP_LENGTH)
    norms = np.linalg.norm(tempogram, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    tempogram = tempogram / norms

    S = tempogram.T @ tempogram
    S = (S - S.min()) / (S.max() - S.min() + 1e-8)

    K    = _checkerboard_kernel(kernel_size)
    half = kernel_size // 2
    N    = S.shape[0]
    novelty = np.zeros(N)
    for t in range(half, N - half):
        block = S[t - half : t + half, t - half : t + half]
        if block.shape == (kernel_size, kernel_size):
            novelty[t] = np.sum(block * K)

    novelty = scipy.ndimage.gaussian_filter1d(novelty, sigma=5)
    min_dist = max(1, int(min_seg_duration * sr / HOP_LENGTH))
    peaks, _ = scipy.signal.find_peaks(
        novelty,
        distance=min_dist,
        height=np.percentile(novelty[novelty > 0], 55),
    )
    boundary_times = librosa.frames_to_time(peaks, sr=sr, hop_length=HOP_LENGTH)
    return _ensure_boundaries(boundary_times, duration)


# ── Algorithm 6: Agglomerative (Homogeneity) ─────────────────────────────────
def segment_agglomerative(
    y: np.ndarray,
    sr: int,
    n_segments: int = 8,
    **_,
) -> np.ndarray:
    """
    Hierarchical agglomerative clustering with Ward linkage and a temporal
    connectivity constraint (librosa.segment.agglomerative).
    Finds internally *homogeneous* segments — opposite objective to novelty
    methods. Beat-synced chroma + MFCC features.
    """
    duration = librosa.get_duration(y=y, sr=sr)

    _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    if len(beats) < 4:
        return _ensure_boundaries(np.array([]), duration)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP_LENGTH, n_mfcc=13)
    bc = librosa.util.sync(chroma, beats, aggregate=np.median)
    bm = librosa.util.sync(mfcc,   beats, aggregate=np.mean)
    features = normalize(np.vstack([bc, bm]))   # (n_features, n_beats)

    n_seg = min(n_segments, max(2, len(beats) // 4))

    # Returns frame indices of boundaries (in beat-frame space)
    bound_frames = librosa.segment.agglomerative(features, n_seg)
    beat_times   = librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LENGTH)

    # Map beat-frame boundaries back to seconds
    boundary_times = []
    for bf in bound_frames:
        if 0 < bf < len(beat_times):
            boundary_times.append(beat_times[bf])

    return _ensure_boundaries(np.array(boundary_times), duration)


# ── Algorithm 7+: madmom downbeat (optional) ─────────────────────────────────
def _segment_madmom(audio_path: Path, bars_per_segment: int = 8) -> Optional[np.ndarray]:
    """
    Uses madmom's deep-learning RNN downbeat tracker to locate bar 1 of each
    bar, then groups every `bars_per_segment` bars into one structural segment.
    Completely different signal from SSM methods: purely rhythmic / metrical.
    """
    try:
        from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
        act    = RNNDownBeatProcessor()(str(audio_path))
        beats  = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)(act)
        # beats columns: [time, beat_number]; beat_number==1 → downbeat
        downbeats = beats[beats[:, 1] == 1, 0]
        if len(downbeats) < 2:
            return None
        # Take every Nth downbeat as a structural boundary
        selected = downbeats[::bars_per_segment]
        duration = float(librosa.get_duration(path=str(audio_path)))
        return _ensure_boundaries(selected, duration)
    except ImportError:
        return None
    except Exception as e:
        print(f"    madmom error: {e}")
        return None


# ── Algorithm 8+: MSAF (optional) ────────────────────────────────────────────
def _segment_msaf(audio_path: Path, algo_id: str) -> Optional[np.ndarray]:
    try:
        import msaf
        boundaries, _ = msaf.process(str(audio_path), boundaries_id=algo_id)
        return _ensure_boundaries(
            np.array(boundaries, dtype=float),
            librosa.get_duration(path=str(audio_path)),
        )
    except ImportError:
        return None
    except Exception as e:
        print(f"    MSAF-{algo_id} error: {e}")
        return None


# ── Algorithm 6: Allin1 (optional, SOTA) ─────────────────────────────────────
def _segment_allin1(audio_path: Path) -> Optional[np.ndarray]:
    import traceback
    try:
        import allin1
    except ImportError:
        return None  # not installed — skip silently

    try:
        print(f"    Running Allin1 on {audio_path.name} (may take 30–60s on CPU)…")
        result = allin1.analyze(str(audio_path))
        segs = result.segments
        if not segs:
            print("    Allin1 returned no segments.")
            return None
        bounds = [s.start for s in segs[1:]] + [segs[-1].end]
        return _ensure_boundaries(np.array(bounds, dtype=float), segs[-1].end)
    except Exception as e:
        print(f"    ✗ Allin1 error: {e}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────
# STEP 3 – CACHE HELPERS
# ─────────────────────────────────────────────

def _cache_path(fname: str) -> Path:
    return CACHE_DIR / f"{fname}.json"


def load_cache(fname: str, audio_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Load cached boundaries for `fname` if they exist and the audio file
    hasn't changed since the cache was written (checked via mtime).
    Returns a dict of {algo_name: np.ndarray} or None on cache miss.
    """
    cp = _cache_path(fname)
    if not cp.exists():
        return None
    try:
        data = json.loads(cp.read_text())
        # Invalidate if the source audio was modified after the cache was written
        if data.get("audio_mtime") != audio_path.stat().st_mtime:
            print(f"  Cache stale for {fname} (audio modified) — recomputing.")
            return None
        results = {
            algo: np.array(bounds, dtype=float)
            for algo, bounds in data["algorithms"].items()
        }
        print(f"  ✓ Loaded from cache ({len(results)} algorithm(s))")
        return results
    except Exception as e:
        print(f"  Cache read error for {fname}: {e} — recomputing.")
        return None


def save_cache(fname: str, audio_path: Path, results: Dict[str, np.ndarray]) -> None:
    """Persist boundaries to processed_cache/{fname}.json."""
    cp = _cache_path(fname)
    data = {
        "audio_mtime": audio_path.stat().st_mtime,
        "algorithms": {
            algo: bounds.tolist()
            for algo, bounds in results.items()
        },
    }
    cp.write_text(json.dumps(data, indent=2))
    print(f"  ✓ Cached to {cp}")


# ─────────────────────────────────────────────
# STEP 4 – RUN PIPELINE
# ─────────────────────────────────────────────

# Core algorithms: always available (no extra installs)
CORE_ALGOS: Dict[str, dict] = {
    "Onset (baseline)": {
        "fn": segment_onset,
        "params": {"min_seg_duration": 4.0},
        "ref": "librosa onset detection",
    },
    "Foote Novelty": {
        "fn": segment_foote,
        "params": {"kernel_size": 64, "min_seg_duration": 3.0},
        "ref": "Foote (2000) — chroma SSM",
    },
    "Foote Tempogram": {
        "fn": segment_foote_tempogram,
        "params": {"kernel_size": 64, "min_seg_duration": 3.0},
        "ref": "Foote (2000) — tempogram SSM (rhythmic breaks)",
    },
    "Agglomerative": {
        "fn": segment_agglomerative,
        "params": {"n_segments": 8},
        "ref": "Ward hierarchical clustering (homogeneity)",
    },
    "Laplacian (McFee'14)": {
        "fn": segment_laplacian,
        "params": {"k": 6, "bandwidth": 0.5},
        "ref": "McFee & Ellis, ISMIR 2014",
    },
    "Spectral Clustering": {
        "fn": segment_spectral,
        "params": {"n_segments": 8},
        "ref": "SSM + sklearn SpectralClustering",
    },
}

# Optional MSAF algorithms (keyed by msaf boundaries_id); 2dftm added
MSAF_ALGOS = ["foote", "scluster", "olda", "cnmf", "2dftm"]


def run_all(audio_path: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray, int]:
    """Load audio and run every available algorithm. Returns (results, y, sr)."""
    y, sr = librosa.load(str(audio_path), sr=SR_TARGET, mono=True)
    results: Dict[str, np.ndarray] = {}

    # Core algorithms
    for name, cfg in CORE_ALGOS.items():
        try:
            b = cfg["fn"](y, sr, **cfg["params"])
            results[name] = b
            print(f"  ✓ {name}: {len(b)-1} segment(s)")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    # madmom downbeat segmentation (optional)
    b = _segment_madmom(audio_path)
    if b is not None:
        results["madmom Downbeat"] = b
        print(f"  ✓ madmom Downbeat: {len(b)-1} segment(s)")

    # MSAF suite (optional)
    for algo_id in MSAF_ALGOS:
        b = _segment_msaf(audio_path, algo_id)
        if b is not None:
            label = f"MSAF-{algo_id.upper()}"
            results[label] = b
            print(f"  ✓ {label}: {len(b)-1} segment(s)")

    # Allin1 (optional, SOTA)
    b = _segment_allin1(audio_path)
    if b is not None:
        results["Allin1 (SOTA)"] = b
        print(f"  ✓ Allin1: {len(b)-1} segment(s)")

    return results, y, sr


# ─────────────────────────────────────────────
# STEP 5 – PRE-COMPUTE ON STARTUP
# ─────────────────────────────────────────────
print("=" * 50)
print("Audio Segmentation Dashboard — startup")
print("=" * 50)
print("\nDownloading audio files…")
download_audio_files()

# ── Optional dependency check ──────────────────
print("\nChecking optional dependencies…")
for pkg, label in [("madmom", "madmom"), ("msaf", "MSAF"), ("allin1", "Allin1")]:
    try:
        __import__(pkg)
        print(f"  ✓ {label} available")
    except ImportError:
        print(f"  – {label} not installed (pip install {pkg})")
# ffmpeg is required by allin1; warn if missing
import shutil
if shutil.which("ffmpeg") is None:
    print("  ⚠ ffmpeg not found — Allin1 needs it: brew install ffmpeg")
print()

ALL_RESULTS: Dict[str, Dict[str, np.ndarray]] = {}
ALL_AUDIO:   Dict[str, Tuple[np.ndarray, int]] = {}


def get_or_compute(fname: str) -> Tuple[Dict[str, np.ndarray], np.ndarray, int]:
    """
    Lazy loader: loads audio and segmentation results on first access,
    then keeps them in memory. Cache is checked before running algorithms.
    """
    path = AUDIO_DIR / f"{fname}.mp3"

    if fname not in ALL_AUDIO:
        print(f"  Loading audio: {fname}…")
        y, sr = librosa.load(str(path), sr=SR_TARGET, mono=True)
        ALL_AUDIO[fname] = (y, sr)

    y, sr = ALL_AUDIO[fname]

    if fname not in ALL_RESULTS:
        cached = load_cache(fname, path)
        if cached is not None:
            ALL_RESULTS[fname] = cached
        else:
            print(f"  Running segmentation: {fname}…")
            res, _, _ = run_all(path)
            ALL_RESULTS[fname] = res
            save_cache(fname, path, res)

    return ALL_RESULTS[fname], y, sr


print("\n✓ Ready — Gradio launching now (files processed on first selection).\n")


# ─────────────────────────────────────────────
# STEP 5 – VISUALISATION HELPERS
# ─────────────────────────────────────────────

def build_plotly_figure(file_name: str) -> go.Figure:
    """Gantt-style segment comparison chart with waveform overlay."""
    results = ALL_RESULTS.get(file_name, {})
    y, sr   = ALL_AUDIO.get(file_name, (np.zeros(100), SR_TARGET))
    duration = librosa.get_duration(y=y, sr=sr)
    n_algos  = len(results)

    fig = go.Figure()

    # ── Waveform (downsampled) ──
    n_pts  = min(8000, len(y))
    times  = np.linspace(0, duration, n_pts)
    y_ds   = np.interp(times, np.linspace(0, duration, len(y)), y)
    # Scale to fit nicely between −0.4 and 0.4 (normalised)
    y_ds  /= max(np.abs(y_ds).max(), 1e-8)
    y_ds  *= 0.38

    fig.add_trace(go.Scatter(
        x=times, y=y_ds,
        mode="lines", name="Waveform",
        line=dict(color="rgba(120,120,140,0.35)", width=0.8),
        yaxis="y2",
        hoverinfo="skip",
    ))

    # ── Segment bars ──
    for algo_idx, (algo_name, boundaries) in enumerate(results.items()):
        color = ALGO_COLORS[algo_idx % len(ALGO_COLORS)]
        y_pos = algo_idx
        n_segs = len(boundaries) - 1

        for i in range(n_segs):
            start, end = float(boundaries[i]), float(boundaries[i + 1])
            # Alternate fill opacity so adjacent segments are distinguishable
            fill_opacity = 0.75 if i % 2 == 0 else 0.45
            r, g, b_ = tuple(int(color.lstrip("#")[j:j+2], 16) for j in (0, 2, 4))

            fig.add_shape(
                type="rect",
                x0=start, x1=end,
                y0=y_pos - 0.40, y1=y_pos + 0.40,
                fillcolor=f"rgba({r},{g},{b_},{fill_opacity})",
                line=dict(color="white", width=0.8),
                layer="above",
            )

            # Duration label (only if segment wide enough)
            if (end - start) / duration > 0.06:
                fig.add_annotation(
                    x=(start + end) / 2, y=y_pos,
                    text=f"{end - start:.0f}s",
                    showarrow=False,
                    font=dict(size=8, color="white"),
                    yref="y",
                )

        # Add a dummy scatter for the legend entry
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=color, symbol="square"),
            name=f"{algo_name} ({n_segs} segs)",
        ))

    # ── Shared boundary dashes ──
    all_b = sorted({t for bs in results.values() for t in bs if 0 < t < duration})
    for t in all_b:
        fig.add_vline(
            x=t, line=dict(color="rgba(60,60,60,0.18)", width=1, dash="dot")
        )

    fig.update_layout(
        title=dict(text=f"Segment Comparison — {file_name}", font=dict(size=15)),
        xaxis=dict(title="Time (seconds)", range=[0, duration], showgrid=True,
                   gridcolor="rgba(200,200,200,0.4)"),
        yaxis=dict(
            tickvals=list(range(n_algos)),
            ticktext=list(results.keys()),
            range=[-0.65, n_algos - 0.35],
            showgrid=False,
        ),
        yaxis2=dict(
            overlaying="y", side="right", showticklabels=False,
            range=[-3.5, 3.5], showgrid=False,
        ),
        height=200 + n_algos * 75,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=190, r=40, t=55, b=120),
    )
    return fig


def build_metrics_df(file_name: str) -> pd.DataFrame:
    """Summary table: #segments, durations, boundary positions."""
    results = ALL_RESULTS.get(file_name, {})
    y, sr   = ALL_AUDIO.get(file_name, (np.zeros(100), SR_TARGET))
    duration = librosa.get_duration(y=y, sr=sr)
    rows = []
    for algo, boundaries in results.items():
        durs = np.diff(boundaries)
        rows.append({
            "Algorithm":      algo,
            "# Segments":     int(len(boundaries) - 1),
            "Avg dur (s)":    f"{durs.mean():.1f}",
            "Min dur (s)":    f"{durs.min():.1f}",
            "Max dur (s)":    f"{durs.max():.1f}",
            "Std dur (s)":    f"{durs.std():.1f}",
            "Boundaries (s)": "  |  ".join(f"{b:.1f}" for b in boundaries[1:-1]),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# STEP 6 – INTERACTIVE PLAYER HTML BUILDER
# ─────────────────────────────────────────────

def build_wavesurfer_html(file_name: str) -> str:
    """
    Self-contained HTML panel:
      • Canvas waveform drawn from downsampled Python audio data
      • Stacked CSS segment rows (one per algorithm)
      • Red playhead that sweeps all rows in sync with the gr.Audio player above
        (attaches a 'timeupdate' listener to Gradio's native <audio> element)
      • Live per-algorithm segment readout
    No WaveSurfer dependency — sync driven purely by the native audio element.
    """
    results  = ALL_RESULTS.get(file_name, {})
    y, sr    = ALL_AUDIO.get(file_name, (np.zeros(100), SR_TARGET))
    duration = float(librosa.get_duration(y=y, sr=sr))

    algo_names      = list(results.keys())
    boundaries_data = {i: [float(b) for b in bs] for i, bs in enumerate(results.values())}
    colors_data     = [ALGO_COLORS[i % len(ALGO_COLORS)] for i in range(len(algo_names))]

    # Downsample waveform to 600 points for canvas (< 6 KB of JSON)
    n_pts  = 600
    y_ds   = np.interp(np.linspace(0, 1, n_pts), np.linspace(0, 1, len(y)), y)
    y_norm = y_ds / max(float(np.abs(y_ds).max()), 1e-8)
    waveform_json = json.dumps([round(float(v), 4) for v in y_norm])

    # Build the stacked rows HTML (pure CSS rectangles, no canvas)
    rows_html = ""
    for idx, (algo, boundaries) in enumerate(results.items()):
        color = colors_data[idx]
        n_segs = len(boundaries) - 1
        segs_html = ""
        for i in range(n_segs):
            start = float(boundaries[i])
            end   = float(boundaries[i + 1])
            left_pct  = start / duration * 100
            width_pct = (end - start) / duration * 100
            opacity   = 0.75 if i % 2 == 0 else 0.45
            label     = f"{end - start:.0f}s" if width_pct > 5 else ""
            r, g, b_  = tuple(int(color.lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
            segs_html += f"""
              <div style="
                position:absolute;
                left:{left_pct:.3f}%;
                width:{width_pct:.3f}%;
                top:2px; bottom:2px;
                background:rgba({r},{g},{b_},{opacity});
                border:1px solid white;
                border-radius:2px;
                display:flex; align-items:center; justify-content:center;
                font-size:10px; color:rgba(255,255,255,0.9);
                overflow:hidden; white-space:nowrap;
              ">{label}</div>"""

        rows_html += f"""
          <div style="display:flex; align-items:center; margin-bottom:4px;">
            <div style="
              width:180px; min-width:180px;
              font-size:11px; font-weight:600;
              color:{color}; padding-right:8px;
              text-align:right; white-space:nowrap; overflow:hidden;
              text-overflow:ellipsis;
            " title="{algo}">{algo}</div>
            <div style="flex:1; position:relative; height:28px; background:#f3f4f6; border-radius:3px; overflow:hidden;">
              {segs_html}
            </div>
          </div>"""

    # Per-algorithm segment readout spans
    seg_spans_html = "".join([
        f"""<span id="seg-{idx}" style="
              display:inline-block; margin:2px;
              padding:3px 9px; border-radius:12px;
              background:{colors_data[idx]}22;
              border:1px solid {colors_data[idx]};
              font-size:11px; color:#333;
            ">{name}: —</span>"""
        for idx, name in enumerate(algo_names)
    ])

    audio_url = f"/file={audio_abs}"

    html = f"""
<div style="font-family:system-ui,sans-serif; padding:12px; background:white; border-radius:8px;">

  <!-- WaveSurfer waveform -->
  <div id="waveform-{file_name}" style="
    border:1px solid #e5e7eb; border-radius:6px;
    background:#f9fafb; margin-bottom:10px; padding:4px;
  "></div>

  <!-- Transport controls -->
  <div style="text-align:center; margin-bottom:12px;">
    <button onclick="window['_wsPlay_{file_name}'] && window['_wsPlay_{file_name}']()"
      style="padding:8px 24px; background:#4F4A85; color:white;
             border:none; border-radius:20px; cursor:pointer; font-size:13px;">
      ▶ Play / Pause
    </button>
    <button onclick="window['_wsStop_{file_name}'] && window['_wsStop_{file_name}']()"
      style="padding:8px 16px; background:#6b7280; color:white;
             border:none; border-radius:20px; cursor:pointer; font-size:13px; margin-left:6px;">
      ■ Stop
    </button>
    <span id="time-{file_name}"
      style="margin-left:14px; font-size:13px; color:#555; font-variant-numeric:tabular-nums;">
      0:00 / 0:00
    </span>
  </div>

  <!-- Stacked segment timeline -->
  <div style="margin-bottom:10px; position:relative;">
    <div style="font-size:11px; color:#9ca3af; margin-bottom:6px; padding-left:188px;">
      Segment timeline — each row is one algorithm
    </div>
    <!-- Playhead overlay (shared across all rows) -->
    <div id="playhead-{file_name}" style="
      position:absolute;
      left:188px; top:22px; bottom:0;
      width:2px; background:rgba(220,38,38,0.8);
      pointer-events:none; z-index:20;
      transition:left 0.05s linear;
    "></div>
    {rows_html}
  </div>

  <!-- Time-axis ruler -->
  <div style="display:flex; margin-left:188px; margin-bottom:10px; position:relative; height:16px;">
    <div id="ruler-{file_name}" style="flex:1; position:relative;"></div>
  </div>

  <!-- Current-segment readout -->
  <div style="font-size:11px; color:#9ca3af; text-align:center; margin-bottom:4px;">
    Current segment (updates during playback):
  </div>
  <div style="text-align:center; line-height:1.8;">
    {seg_spans_html}
  </div>
</div>

<script src="https://unpkg.com/wavesurfer.js@6.6.4/dist/wavesurfer.min.js"></script>
<script>
(function() {{
  // ── Constants injected from Python ──
  var FILE     = {json.dumps(file_name)};
  var DURATION = {duration:.4f};
  var BOUNDS   = {json.dumps(boundaries_data)};
  var NAMES    = {json.dumps(algo_names)};
  var AUDIO_URL = {json.dumps(audio_url)};

  function fmtTime(t) {{
    var m = Math.floor(t / 60);
    var s = String(Math.floor(t % 60)).padStart(2, '0');
    return m + ':' + s;
  }}

  // ── Time-axis ruler ──
  var ruler = document.getElementById('ruler-' + FILE);
  if (ruler && DURATION > 0) {{
    var step = Math.max(10, Math.ceil(DURATION / 10 / 10) * 10);
    for (var t = 0; t <= DURATION; t += step) {{
      var el = document.createElement('span');
      var mm = String(Math.floor(t / 60)).padStart(2, '0');
      var ss = String(Math.floor(t % 60)).padStart(2, '0');
      el.textContent = mm + ':' + ss;
      el.style.cssText = 'position:absolute;left:' + (t/DURATION*100).toFixed(2) +
        '%;transform:translateX(-50%);font-size:9px;color:#9ca3af;';
      ruler.appendChild(el);
    }}
  }}

  // ── WaveSurfer (v6 UMD — executes reliably in Gradio's HTML component) ──
  var ws = WaveSurfer.create({{
    container: '#waveform-' + FILE,
    waveColor:     '#a5b4fc',
    progressColor: '#4F4A85',
    height: 72,
    normalize: true,
    interact: true,
    backend: 'WebAudio',
  }});

  ws.load(AUDIO_URL);
  window['_ws_' + FILE] = ws;

  ws.on('ready', function() {{
    var el = document.getElementById('time-' + FILE);
    if (el) el.textContent = '0:00 / ' + fmtTime(ws.getDuration());
  }});

  ws.on('error', function(err) {{
    console.warn('WaveSurfer error:', err);
    var el = document.getElementById('waveform-' + FILE);
    if (el) el.innerHTML =
      '<p style="color:#f87171;padding:8px;font-size:12px;">⚠ Waveform failed to load — use the Audio Player tab above instead.</p>';
  }});

  ws.on('audioprocess', function(t) {{
    // time display
    var timeEl = document.getElementById('time-' + FILE);
    if (timeEl) timeEl.textContent = fmtTime(t) + ' / ' + fmtTime(ws.getDuration());

    // playhead
    var ph = document.getElementById('playhead-' + FILE);
    if (ph) {{
      var parent = ph.parentElement;
      var rowWidth = parent ? parent.getBoundingClientRect().width - 188 : 0;
      var pct = DURATION > 0 ? t / DURATION : 0;
      ph.style.left = (188 + pct * rowWidth) + 'px';
    }}

    // segment readouts
    Object.keys(BOUNDS).forEach(function(idx) {{
      var bounds = BOUNDS[idx];
      var segIdx = 0;
      for (var i = 0; i < bounds.length - 1; i++) {{
        if (t >= bounds[i] && t < bounds[i + 1]) {{ segIdx = i; break; }}
      }}
      var el = document.getElementById('seg-' + idx);
      if (el && bounds.length > 1) {{
        var end = bounds[segIdx + 1] !== undefined ? bounds[segIdx + 1] : DURATION;
        el.textContent = NAMES[idx] + ': Seg ' + (segIdx + 1) +
          ' (' + bounds[segIdx].toFixed(1) + '–' + end.toFixed(1) + 's)';
      }}
    }});
  }});

  // Expose play/pause/stop to inline onclick buttons
  window['_wsPlay_'  + FILE] = function() {{ ws.playPause(); }};
  window['_wsStop_'  + FILE] = function() {{ ws.stop(); }};
}})();
</script>
"""
    return html


# ─────────────────────────────────────────────
# STEP 7 – GRADIO UI
# ─────────────────────────────────────────────

def on_select(file_name: str):
    audio_path = str(AUDIO_DIR / f"{file_name}.mp3")
    ws_html    = build_wavesurfer_html(file_name)
    fig        = build_plotly_figure(file_name)
    metrics    = build_metrics_df(file_name)
    return audio_path, ws_html, fig, metrics


file_names = list(AUDIO_FILES.keys())

with gr.Blocks(title="Audio Segmentation Comparison", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# 🎵 Audio Segmentation — Algorithm Comparison

Compare SOTA structural segmentation algorithms for sequential AI.
Select a track; all algorithms run on startup so switching is instant.
""")

    with gr.Row():
        file_dd = gr.Dropdown(
            choices=file_names,
            value=file_names[0],
            label="Audio file",
            scale=3,
        )
        gr.Markdown(
            "**Built-in:** Onset · Foote Novelty · Laplacian (McFee'14) · Spectral Clustering  \n"
            "**Optional (install separately):** MSAF suite · Allin1 (SOTA transformer)",
            scale=4,
        )

    # Always-reliable audio player at the top
    audio_player = gr.Audio(
        label="Audio Player",
        type="filepath",
    )

    with gr.Tabs():
        with gr.Tab("🎧 Interactive Playback"):
            gr.Markdown(
                "_Waveform is clickable for seeking. "
                "The red playhead and segment readouts update in real-time. "
                "If the waveform shows an error, use the Audio Player above._"
            )
            ws_output = gr.HTML()

        with gr.Tab("📊 Segment Timeline"):
            plot_output = gr.Plot()

        with gr.Tab("📋 Metrics"):
            metrics_output = gr.DataFrame(wrap=True)

    # Wire up events
    outputs = [audio_player, ws_output, plot_output, metrics_output]
    demo.load(fn=on_select, inputs=[file_dd], outputs=outputs)
    file_dd.change(fn=on_select, inputs=[file_dd], outputs=outputs)


if __name__ == "__main__":
    demo.launch(
        allowed_paths=[str(AUDIO_DIR.resolve())],
        share=False,
        server_port=7860,
    )
