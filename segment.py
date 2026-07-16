#!/usr/bin/env python3
"""
segment.py — Audio segmentation pre-processor
==============================================
Runs all available SOTA segmentation algorithms on the configured audio files
and writes results.json (read by viewer.html).

Usage:
    python segment.py            # process all files
    python segment.py --force    # ignore cache, recompute everything

Then open the dashboard:
    python -m http.server 8080
    # open http://localhost:8080/viewer.html

Optional extras (install for more algorithms):
    pip install madmom           # downbeat-based segmentation
    pip install msaf             # MSAF suite (Foote, SCLUSTER, OLDA, CNMF, 2DFT)
    pip install allin1 torch     # transformer SOTA (Kim et al. 2023)
    brew install ffmpeg          # required by allin1
"""

import argparse
import json
import traceback
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import requests
import scipy.linalg
import scipy.ndimage
import scipy.signal
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

AUDIO_DIR  = Path("./audio");   AUDIO_DIR.mkdir(exist_ok=True)
CACHE_DIR  = Path("./processed_cache"); CACHE_DIR.mkdir(exist_ok=True)
RESULTS_FILE = Path("./results.json")

HOP_LENGTH = 512
SR_TARGET  = 22050
WAVEFORM_PTS = 800   # points in the downsampled waveform sent to viewer


# ─────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────
def download_audio_files() -> None:
    for name, url in AUDIO_FILES.items():
        path = AUDIO_DIR / f"{name}.mp3"
        if not path.exists():
            print(f"  Downloading {name}…", end=" ", flush=True)
            try:
                r = requests.get(url, timeout=30); r.raise_for_status()
                path.write_bytes(r.content); print("✓")
            except Exception as e:
                print(f"✗ ({e})")


# ─────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────
def _cache_path(fname: str) -> Path:
    return CACHE_DIR / f"{fname}.json"

def load_cache(fname: str, audio_path: Path) -> Optional[Dict]:
    cp = _cache_path(fname)
    if not cp.exists():
        return None
    try:
        data = json.loads(cp.read_text())
        if data.get("audio_mtime") != audio_path.stat().st_mtime:
            print(f"  Cache stale ({fname}) — recomputing.")
            return None
        return data
    except Exception:
        return None

def save_cache(fname: str, audio_path: Path, data: dict) -> None:
    data["audio_mtime"] = audio_path.stat().st_mtime
    _cache_path(fname).write_text(json.dumps(data, indent=2))


# ─────────────────────────────────────────────
# ALGORITHM HELPERS
# ─────────────────────────────────────────────
def _ensure_boundaries(b: np.ndarray, duration: float) -> np.ndarray:
    b = np.unique(np.round(np.concatenate([[0.0], b, [duration]]), 4))
    return b[(b >= 0) & (b <= duration + 0.001)]

def _checkerboard_kernel(L: int) -> np.ndarray:
    half = L // 2
    g = scipy.signal.windows.gaussian(L, std=L / 8)
    K = np.outer(g, g)
    checker = np.ones((L, L))
    checker[:half, half:] = -1
    checker[half:, :half] = -1
    return K * checker


# ─────────────────────────────────────────────
# ALGORITHMS
# ─────────────────────────────────────────────
def seg_onset(y, sr, min_seg_duration=4.0, **_):
    """Librosa onset detection (baseline)."""
    t = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH, units="time")
    dur = librosa.get_duration(y=y, sr=sr)
    filt = [0.0]
    for v in t:
        if v - filt[-1] >= min_seg_duration:
            filt.append(float(v))
    return _ensure_boundaries(np.array(filt[1:]), dur)


def _foote_core(features: np.ndarray, sr: int, kernel_size: int,
                min_seg_duration: float, y_len: int) -> np.ndarray:
    norms = np.linalg.norm(features, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    F = features / norms
    S = F.T @ F
    S = (S - S.min()) / (S.max() - S.min() + 1e-8)
    K = _checkerboard_kernel(kernel_size)
    half = kernel_size // 2
    N = S.shape[0]
    novelty = np.zeros(N)
    for t in range(half, N - half):
        block = S[t - half: t + half, t - half: t + half]
        if block.shape == (kernel_size, kernel_size):
            novelty[t] = np.sum(block * K)
    novelty = scipy.ndimage.gaussian_filter1d(novelty, sigma=5)
    min_dist = max(1, int(min_seg_duration * sr / HOP_LENGTH))
    peaks, _ = scipy.signal.find_peaks(
        novelty, distance=min_dist,
        height=np.percentile(novelty[novelty > 0], 55) if novelty.max() > 0 else 0
    )
    dur = y_len / sr
    return _ensure_boundaries(librosa.frames_to_time(peaks, sr=sr, hop_length=HOP_LENGTH), dur)


def seg_foote(y, sr, kernel_size=64, min_seg_duration=3.0, **_):
    """Foote Novelty on chroma SSM (harmonic/timbral breaks)."""
    return _foote_core(
        librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH),
        sr, kernel_size, min_seg_duration, len(y)
    )


def seg_foote_tempogram(y, sr, kernel_size=64, min_seg_duration=3.0, **_):
    """Foote Novelty on tempogram SSM (rhythmic breaks)."""
    return _foote_core(
        librosa.feature.tempogram(y=y, sr=sr, hop_length=HOP_LENGTH),
        sr, kernel_size, min_seg_duration, len(y)
    )


def seg_agglomerative(y, sr, n_segments=8, **_):
    """Ward hierarchical clustering — finds homogeneous regions."""
    dur = librosa.get_duration(y=y, sr=sr)
    _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    if len(beats) < 4:
        return _ensure_boundaries(np.array([]), dur)
    k = min(n_segments, max(2, len(beats) // 4))
    bc = librosa.util.sync(librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH), beats, aggregate=np.median)
    bm = librosa.util.sync(librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP_LENGTH, n_mfcc=13), beats, aggregate=np.mean)
    features = normalize(np.vstack([bc, bm]))
    bound_frames = librosa.segment.agglomerative(features, k)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LENGTH)
    bt = [beat_times[bf] for bf in bound_frames if 0 < bf < len(beat_times)]
    return _ensure_boundaries(np.array(bt), dur)


def seg_laplacian(y, sr, k=8, bandwidth=0.5, min_seg_duration=10.0, **_):
    """Laplacian segmentation (McFee & Ellis, ISMIR 2014)."""
    dur = librosa.get_duration(y=y, sr=sr)
    _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    if len(beats) < 4:
        return _ensure_boundaries(np.array([]), dur)
    k = min(k, max(2, len(beats) // 4))
    bc = librosa.util.sync(librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH), beats, aggregate=np.median)
    bm = librosa.util.sync(librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP_LENGTH, n_mfcc=13), beats, aggregate=np.mean)
    features = normalize(np.vstack([bc, bm]).T)
    R = librosa.segment.recurrence_matrix(features.T, mode="affinity", sym=True, bandwidth=bandwidth)
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    R  = df(R, size=(1, 7))
    R  = librosa.segment.path_enhance(R, 15, window="hann", n_filters=7)
    deg = R.sum(axis=1); deg[deg == 0] = 1.0
    D_inv = np.diag(1.0 / np.sqrt(deg))
    Ln = np.eye(len(deg)) - D_inv @ R @ D_inv
    try:
        _, vecs = scipy.linalg.eigh(Ln, subset_by_index=[0, k])
    except Exception:
        _, vecs = np.linalg.eigh(Ln); vecs = vecs[:, :k + 1]
    embedding = normalize(vecs[:, 1:])
    seg_ids = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(embedding)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LENGTH)
    # Raw boundaries: every beat where cluster label changes
    raw = [beat_times[i] for i in range(1, len(seg_ids))
           if seg_ids[i] != seg_ids[i - 1] and i < len(beat_times)]
    # Suppress boundaries closer than min_seg_duration to the previous one
    merged, last = [], -min_seg_duration
    for t in raw:
        if t - last >= min_seg_duration:
            merged.append(t)
            last = t
    return _ensure_boundaries(np.array(merged), dur)


def seg_spectral(y, sr, n_segments=8, **_):
    """Spectral clustering on frame-level SSM."""
    dur = librosa.get_duration(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP_LENGTH, n_mfcc=13)
    features = normalize(np.vstack([chroma, mfcc]).T)
    step = max(1, len(features) // 300)
    fd = features[::step]
    S = np.clip(fd @ fd.T, 0, 1); np.fill_diagonal(S, 0)
    n_seg = min(n_segments, max(2, len(fd) // 8))
    labels = SpectralClustering(n_clusters=n_seg, affinity="precomputed", n_init=5, random_state=42).fit_predict(S + 1e-8)
    from scipy.stats import mode as scipy_mode
    smoothed = np.array([int(scipy_mode(labels[max(0, i-3):i+4], keepdims=False).mode) for i in range(len(labels))])
    bt = []
    for i in range(1, len(smoothed)):
        if smoothed[i] != smoothed[i - 1]:
            t = (i * step) * HOP_LENGTH / sr
            if 0.0 < t < dur:
                bt.append(t)
    return _ensure_boundaries(np.array(bt), dur)


def seg_madmom(audio_path: Path, bars_per_segment=8) -> Optional[np.ndarray]:
    try:
        _patch_numpy_aliases()
        _patch_collections()
        _patch_pkg_resources()
        from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
        act    = RNNDownBeatProcessor()(str(audio_path))
        beats  = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)(act)
        downbeats = beats[beats[:, 1] == 1, 0]
        if len(downbeats) < 2:
            return None
        dur = float(librosa.get_duration(path=str(audio_path)))
        return _ensure_boundaries(downbeats[::bars_per_segment], dur)
    except ImportError:
        return None
    except Exception as e:
        print(f"    madmom error: {e}"); return None


def _patch_numpy_aliases():
    """madmom uses np.float, np.int, np.complex etc. which were removed in NumPy 1.24.
    Re-inject them as aliases for the builtin types."""
    import numpy as np, warnings
    for name, builtin in [('float', float), ('int', int),
                           ('complex', complex), ('bool', bool)]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if not hasattr(np, name):
                setattr(np, name, builtin)


def _patch_collections():
    """madmom 0.16.1 uses `from collections import MutableSequence` etc.,
    which were removed from collections in Python 3.10 (moved to collections.abc).
    Re-inject them so madmom's Cython extensions can import."""
    import collections, collections.abc
    _moved = [
        'Callable', 'Iterable', 'Iterator', 'Generator',
        'Mapping', 'MutableMapping', 'MappingView',
        'KeysView', 'ItemsView', 'ValuesView',
        'Sequence', 'MutableSequence',
        'Set', 'MutableSet',
        'Hashable', 'Sized', 'Container',
    ]
    for name in _moved:
        if not hasattr(collections, name) and hasattr(collections.abc, name):
            setattr(collections, name, getattr(collections.abc, name))


def _patch_pkg_resources():
    """madmom uses pkg_resources for version info. On some Python 3.11+ venvs
    the module is missing even when setuptools is installed. Inject a minimal shim."""
    try:
        import pkg_resources  # noqa: F401
        return  # already available
    except ImportError:
        pass
    import sys, types
    pkg = types.ModuleType('pkg_resources')
    class _Dist:
        def __init__(self): self.version = '0.0.0'
    pkg.get_distribution   = lambda name: _Dist()
    pkg.parse_version      = lambda v: tuple(int(x) for x in str(v).split('.')[:3] if x.isdigit())
    pkg.resource_filename  = lambda pkg_name, path: path
    pkg.resource_listdir   = lambda pkg_name, path: []
    pkg.resource_string    = lambda pkg_name, path: b''
    sys.modules['pkg_resources'] = pkg


def _patch_scipy_inf():
    """MSAF (v0.1.80) uses `from scipy import inf` which was removed in scipy 1.8.
    Inject it back onto the scipy module so msaf's import succeeds."""
    import scipy
    if not hasattr(scipy, 'inf'):
        scipy.inf = float('inf')


def seg_msaf(audio_path: Path, algo_id: str) -> Optional[np.ndarray]:
    import importlib.util
    if importlib.util.find_spec("msaf") is None:
        return None  # not installed
    try:
        _patch_scipy_inf()
        import msaf
        boundaries, _ = msaf.process(str(audio_path), boundaries_id=algo_id)
        dur = librosa.get_duration(path=str(audio_path))
        return _ensure_boundaries(np.array(boundaries, dtype=float), dur)
    except Exception as e:
        print(f"    MSAF-{algo_id} error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def seg_allin1(audio_path: Path) -> Optional[np.ndarray]:
    import importlib.util
    if importlib.util.find_spec("allin1") is None:
        return None  # not installed
    try:
        _patch_numpy_aliases()
        _patch_collections()
        _patch_pkg_resources()
        import allin1
    except Exception as e:
        print(f"    Allin1 import error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None
    try:
        print(f"    Allin1 running on {audio_path.name} (may take ~60s on CPU)…")
        result = allin1.analyze(str(audio_path))
        segs = result.segments
        if not segs:
            return None
        bounds = [s.start for s in segs[1:]] + [segs[-1].end]
        return _ensure_boundaries(np.array(bounds, dtype=float), segs[-1].end)
    except Exception as e:
        print(f"    Allin1 error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────
# HYBRID: Allin1 + Agglomerative
# ─────────────────────────────────────────────

def seg_hybrid(allin1_bounds: list, agglom_bounds: list, duration: float,
               bpm: Optional[float] = None,
               long_thresh: float = 30.0,
               tiny_beats: float = 32.0,
               snap_radius: float = 15.0) -> Optional[np.ndarray]:
    """Refine Allin1 boundaries using Agglomerative's 'tiny section' pattern.

    Observation: Agglomerative often produces one very long span followed by a
    tiny transitional section. The *end* of that tiny section is a reliable
    structural boundary even when Allin1 places its nearest boundary slightly
    off. This function snaps the closest Allin1 boundary to the Agglomerative
    candidate, or inserts it if no Allin1 boundary is nearby.

    The 'tiny' threshold is tempo-relative: a section counts as tiny when it is
    at most `tiny_beats` beats long, i.e. tiny_thresh = tiny_beats * 60 / bpm.
    32 beats ≈ 8 bars in 4/4 — beyond that a section is too long to be a mere
    transition. If bpm is unavailable, it falls back to a fixed 8.0 s.

    Args:
        bpm:         song tempo in BPM (drives the tiny threshold)
        long_thresh: segment duration (s) to qualify as 'long'
        tiny_beats:  max length in BEATS for a section to count as 'tiny'
        snap_radius: max distance (s) to snap an Allin1 boundary to a candidate
    """
    if not allin1_bounds or not agglom_bounds:
        return None

    # Tempo-relative 'tiny' threshold (seconds); fall back to 8.0 s w/o tempo.
    if bpm and bpm > 0:
        tiny_thresh = tiny_beats * 60.0 / bpm
    else:
        tiny_thresh = 8.0

    a1 = np.array(allin1_bounds)
    ag = np.array(agglom_bounds)
    ag_durs = np.diff(ag)

    # Find tiny sections preceded by a long section
    candidates = []
    for i in range(1, len(ag_durs)):
        if ag_durs[i - 1] >= long_thresh and ag_durs[i] <= tiny_thresh:
            # End of the tiny section = reliable boundary
            candidates.append(float(ag[i + 1]))

    if not candidates:
        # No pattern found — return Allin1 unchanged
        return _ensure_boundaries(a1, duration)

    # Inner Allin1 boundaries (exclude 0.0 and duration)
    result = list(a1[1:-1])

    for cand in candidates:
        if not result:
            result.append(cand)
            continue
        dists = [abs(cand - b) for b in result]
        min_dist = min(dists)
        if min_dist <= snap_radius:
            # Snap: replace the nearest Allin1 boundary
            result[dists.index(min_dist)] = cand
        else:
            # Insert: Allin1 has no boundary nearby — add it
            result.append(cand)

    result = sorted(set(round(t, 4) for t in result))
    tag = f"tiny≤{tiny_thresh:.1f}s" + (f" @ {bpm:.0f} BPM" if bpm else " (fixed)")
    print(f"    Hybrid: {len(candidates)} Agglom candidate(s) applied ({tag})")
    return _ensure_boundaries(np.array(result), duration)


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────
CORE_ALGOS = {
    "Onset (baseline)":    (seg_onset,           {"min_seg_duration": 4.0}),
    "Foote Novelty":       (seg_foote,            {"kernel_size": 64, "min_seg_duration": 3.0}),
    "Foote Tempogram":     (seg_foote_tempogram,  {"kernel_size": 64, "min_seg_duration": 3.0}),
    "Agglomerative":       (seg_agglomerative,    {"n_segments": 8}),
    "Laplacian (McFee'14)":(seg_laplacian,        {"k": 8, "bandwidth": 0.5, "min_seg_duration": 10.0}),
    "Spectral Clustering": (seg_spectral,         {"n_segments": 8}),
}
MSAF_IDS = ["foote", "scluster", "olda", "cnmf", "2dftm"]


def _tempo_bpm(y, sr) -> float:
    """Global tempo (BPM) estimate via librosa beat tracking."""
    t = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    return float(np.atleast_1d(t)[0])


def _ensure_tempo(cached: Optional[dict], audio_path: Path) -> float:
    """Return the cached tempo, or compute it from the audio if not cached."""
    bpm = (cached or {}).get("tempo")
    if bpm:
        return float(bpm)
    y, sr = librosa.load(str(audio_path), sr=SR_TARGET, mono=True)
    return _tempo_bpm(y, sr)


def _compute_hybrid(algos: dict, duration: float, bpm: Optional[float] = None) -> None:
    """Compute Hybrid (Allin1+Agglom) in-place into algos dict, if possible."""
    if "Allin1 (SOTA)" in algos and "Agglomerative" in algos:
        b = seg_hybrid(algos["Allin1 (SOTA)"], algos["Agglomerative"], duration, bpm=bpm)
        if b is not None:
            algos["Hybrid (Allin1+Agglom)"] = [round(float(v), 4) for v in b]
            print(f"  ✓ Hybrid (Allin1+Agglom): {len(b)-1} segment(s)")
    else:
        missing = [k for k in ("Allin1 (SOTA)", "Agglomerative") if k not in algos]
        print(f"  – Hybrid skipped (missing: {', '.join(missing)})")


def process_file(fname: str, audio_path: Path, force: bool,
                 hybrid_only: bool = False) -> dict:
    """Load audio, run all algorithms, return result dict (cache-aware).

    hybrid_only: skip all base algorithms; only (re-)compute the Hybrid row
                 from already-cached Allin1 and Agglomerative results.
    """
    cached = load_cache(fname, audio_path)

    if hybrid_only:
        if cached and "algorithms" in cached:
            algos    = cached["algorithms"]
            waveform = cached.get("waveform", [])
            duration = cached.get("duration", 0.0)
            bpm      = _ensure_tempo(cached, audio_path)
            print(f"  (hybrid-only) updating from cache… ({bpm:.1f} BPM)")
            _compute_hybrid(algos, duration, bpm)
            save_cache(fname, audio_path,
                       {"duration": duration, "tempo": bpm,
                        "waveform": waveform, "algorithms": algos})
        else:
            print(f"  ✗ No cache found — run without --hybrid-only first")
            algos, waveform, duration, bpm = {}, [], 0.0, None
        return {"duration": duration, "tempo": bpm,
                "waveform": waveform, "algorithms": algos}

    if not force and cached and "algorithms" in cached:
        print(f"  ✓ Loaded from cache")
        algos    = cached["algorithms"]
        waveform = cached.get("waveform", [])
        duration = cached.get("duration", 0.0)
        bpm      = _ensure_tempo(cached, audio_path)
        # Always refresh the hybrid in case parameters changed
        _compute_hybrid(algos, duration, bpm)
        save_cache(fname, audio_path,
                   {"duration": duration, "tempo": bpm,
                    "waveform": waveform, "algorithms": algos})
    else:
        y, sr = librosa.load(str(audio_path), sr=SR_TARGET, mono=True)
        duration = float(librosa.get_duration(y=y, sr=sr))
        bpm = _tempo_bpm(y, sr)
        print(f"  tempo: {bpm:.1f} BPM")

        # Downsampled waveform for viewer
        y_ds = np.interp(np.linspace(0, 1, WAVEFORM_PTS), np.linspace(0, 1, len(y)), y)
        y_ds /= max(float(np.abs(y_ds).max()), 1e-8)
        waveform = [round(float(v), 4) for v in y_ds]

        algos = {}
        for name, (fn, params) in CORE_ALGOS.items():
            try:
                b = fn(y, sr, **params)
                algos[name] = [round(float(v), 4) for v in b]
                print(f"  ✓ {name}: {len(b)-1} segment(s)")
            except Exception as e:
                print(f"  ✗ {name}: {e}")

        # madmom
        b = seg_madmom(audio_path)
        if b is not None:
            algos["madmom Downbeat"] = [round(float(v), 4) for v in b]
            print(f"  ✓ madmom Downbeat: {len(b)-1} segment(s)")

        # MSAF
        for algo_id in MSAF_IDS:
            b = seg_msaf(audio_path, algo_id)
            if b is not None:
                label = f"MSAF-{algo_id.upper()}"
                algos[label] = [round(float(v), 4) for v in b]
                print(f"  ✓ {label}: {len(b)-1} segment(s)")

        # Allin1
        b = seg_allin1(audio_path)
        if b is not None:
            algos["Allin1 (SOTA)"] = [round(float(v), 4) for v in b]
            print(f"  ✓ Allin1: {len(b)-1} segment(s)")

        # Hybrid (depends on Allin1 + Agglomerative)
        _compute_hybrid(algos, duration, bpm)

        save_cache(fname, audio_path,
                   {"duration": duration, "tempo": bpm,
                    "waveform": waveform, "algorithms": algos})

    return {"duration": duration, "tempo": bpm,
            "waveform": waveform, "algorithms": algos}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Ignore cache, recompute all")
    parser.add_argument("--hybrid-only", action="store_true",
                        help="Only (re-)compute Hybrid (Allin1+Agglom) from cached data; "
                             "skip all other algorithms")
    args = parser.parse_args()

    print("=" * 50)
    print("segment.py — Audio Segmentation Pre-processor")
    print("=" * 50)

    print("\nChecking optional dependencies…")
    import shutil, importlib.util, traceback as _tb
    for pkg, label in [("madmom", "madmom"), ("msaf", "MSAF"), ("allin1", "Allin1")]:
        spec = importlib.util.find_spec(pkg)
        if spec is None:
            print(f"  – {label} not installed  (pip install {pkg})")
            continue
        # Package is installed — try actually importing it
        try:
            if pkg == "msaf":
                _patch_scipy_inf()   # scipy.inf removed in 1.8, needed by msaf
            if pkg in ("madmom", "allin1"):
                _patch_numpy_aliases()  # np.float etc. removed in NumPy 1.24
                _patch_collections()    # collections.MutableSequence removed in 3.10
                _patch_pkg_resources()  # pkg_resources missing in some venvs
            __import__(pkg)
            print(f"  ✓ {label} ({spec.origin})")
        except Exception as e:
            print(f"  ⚠ {label} installed but fails to import: {e}")
            _tb.print_exc()
    if shutil.which("ffmpeg") is None:
        print("  ⚠ ffmpeg not found — needed by Allin1 (brew install ffmpeg)")

    print("\nDownloading audio files…")
    download_audio_files()

    all_data = {}
    for fname, url in AUDIO_FILES.items():
        path = AUDIO_DIR / f"{fname}.mp3"
        if not path.exists():
            print(f"\n{fname}: file missing, skipping")
            continue
        print(f"\n{fname}:")
        all_data[fname] = {
            "url": f"audio/{fname}.mp3",
            **process_file(fname, path, args.force,
                           hybrid_only=args.hybrid_only),
        }

    RESULTS_FILE.write_text(json.dumps(all_data, indent=2))
    print(f"\n✓ Results written to {RESULTS_FILE}")
    print("\nStart the dashboard:")
    print("  python -m http.server 8080")
    print("  open http://localhost:8080/viewer.html")


if __name__ == "__main__":
    main()
