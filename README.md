# Audio Segmentation Comparison Dashboard

Compares SOTA structural segmentation algorithms across multiple audio files with
interactive waveform playback and real-time segment visualisation. Built to help
choose the best algorithm for use in sequential AI pipelines.

---

## Two-step workflow (recommended)

### Step 1 — Teacher pre-computes (once)

**Core algorithms only** (fast, no extra installs):

```bash
# Use Python 3.11 — required for madmom/allin1 compatibility
~/.pyenv/versions/3.11.9/bin/python -m venv segmentation_env_311
source segmentation_env_311/bin/activate

pip install -r requirements.txt
python segment.py          # downloads audio + runs all 6 core algorithms
python segment.py --force  # recompute everything (ignore cache)
```

**To also include MSAF + madmom + Allin1** (SOTA semantic segmentation):

```bash
# MSAF
pip install msaf

# madmom (required by Allin1; needs Cython to build)
pip install cython numpy
pip install madmom --no-build-isolation
pip install --force-reinstall setuptools   # fixes pkg_resources import errors

# Allin1 + torch
pip install torch
pip install allin1

# Patch Allin1 to work with natten >= 0.15 (API was renamed)
# — already done: see allin1/models/dinat.py in site-packages
# If you reinstall allin1, re-apply the patch from the project repo.

python segment.py --force  # recompute with all algorithms
```

> **Python version note:** use Python 3.11. Python 3.12 breaks madmom's
> Cython extensions. Python 3.10+ breaks `collections.MutableSequence`
> (patched automatically by `segment.py`).

> **Allin1 note:** downloads ~2 GB model weights on first run. Subsequent
> runs use the cache. GPU optional; CPU works fine for 6 tracks.

Audio files land in `audio/`, per-file caches in `processed_cache/`, and the
viewer reads from `results.json` in the project root.

### Step 2 — Students view (no Python setup needed)

Share the whole folder (or a zip) with students. They only need to run:

```bash
# From the segmentation folder:
python -m http.server 8080
```

Then open **http://localhost:8080/viewer.html** in any browser.

> Everyone has Python — no pip installs, no virtual environments, no Gradio.

The cache is automatically invalidated if an audio file is replaced (detected
via modification time). Re-run `python segment.py` to refresh.

---

## Algorithms

### Built-in (no extra install needed)

| Algorithm | Reference | Notes |
|---|---|---|
| **Onset (baseline)** | librosa onset detection | Fast; over-segments on dense music |
| **Foote Novelty** | Foote (2000) | Checkerboard kernel on chroma SSM; detects harmonic/timbral breaks |
| **Foote Tempogram** | Foote (2000) | Same kernel, tempogram SSM; detects *rhythmic* breaks instead |
| **Agglomerative** | Ward hierarchical clustering | Finds internally *homogeneous* segments; opposite objective to novelty methods |
| **Laplacian** | McFee & Ellis, ISMIR 2014 | Beat-synced graph Laplacian + k-means; strong on repetitive structure |
| **Spectral Clustering** | SSM + sklearn | Frame-level affinity matrix; flexible but slower |

### Optional — madmom downbeat segmentation

> **Not recommended for semantic segmentation.** madmom groups bars by downbeat
> (rhythmic/metrical) — no concept of verse vs. chorus. It is however a
> **required dependency of Allin1**, so install it if you want Allin1.

madmom requires Cython and does not install via plain `pip install`. Use:

```bash
pip install cython numpy
pip install madmom --no-build-isolation
pip install --force-reinstall setuptools   # fixes pkg_resources errors
```

Adds: **madmom Downbeat**

### Optional — MSAF suite

MSAF bundles several peer-reviewed structural segmentation algorithms.

```bash
pip install msaf
```

Adds: **MSAF-FOOTE**, **MSAF-SCLUSTER**, **MSAF-OLDA**, **MSAF-CNMF**, **MSAF-2DFTM**

> MSAF uses `from scipy import inf` which was removed in scipy 1.8.
> `segment.py` patches this automatically — no action needed.

### Optional — Allin1 (current SOTA, semantic)

Transformer-based analyser (Kim et al., ISMIR 2023). Uses Demucs source
separation + a transformer trained on RWC + Harmonix. The only algorithm here
that produces **labelled** sections (verse, chorus, bridge, etc.).

```bash
pip install torch
pip install cython numpy
pip install madmom --no-build-isolation    # Allin1 runtime dependency
pip install --force-reinstall setuptools
pip install allin1
```

**Compatibility issues fixed automatically by `segment.py`:**
- `collections.MutableSequence` removed in Python 3.10 → patched
- `np.float` removed in NumPy 1.24 → patched
- `pkg_resources` missing in some venvs → patched

**One manual patch required** (natten API renamed in v0.15+):
After installing allin1, edit
`segmentation_env_311/lib/python3.11/site-packages/allin1/models/dinat.py`
line 10 — replace the bare import with the try/except shim already present
in the copy of that file in this project (or copy it directly from here).

> Downloads ~2 GB model weights on first run (cached after that).
> CPU is fine for 6 tracks.

---

## Tuning parameters

All tunable knobs are in the `CORE_ALGOS` dict near the top of `segment.py`.
Current values and the reasoning behind each choice:

```python
CORE_ALGOS = {
    "Onset (baseline)":     (seg_onset,          {"min_seg_duration": 4.0}),
    "Foote Novelty":        (seg_foote,           {"kernel_size": 64, "min_seg_duration": 3.0}),
    "Foote Tempogram":      (seg_foote_tempogram, {"kernel_size": 64, "min_seg_duration": 3.0}),
    "Agglomerative":        (seg_agglomerative,   {"n_segments": 8}),
    "Laplacian (McFee'14)": (seg_laplacian,       {"k": 8, "bandwidth": 0.5, "min_seg_duration": 10.0}),
    "Spectral Clustering":  (seg_spectral,        {"n_segments": 8}),
}
```

### Parameter decisions

**Onset — `min_seg_duration: 4.0 s`**
Onset detection fires on every transient (attack), producing hundreds of raw
boundaries. The 4 s floor suppresses micro-segments from drum hits or ornaments
that are too short to be structural sections.

**Foote Novelty / Foote Tempogram — `kernel_size: 64`, `min_seg_duration: 3.0 s`**
`kernel_size` sets how many frames the checkerboard kernel spans. At the default
hop length (512 samples, 22 kHz), 64 frames ≈ 1.5 s of context per side — enough
to catch verse/chorus-scale transitions without being too local. A larger kernel
(e.g. 128) responds to longer-range changes but may miss shorter sections. The
3 s floor is slightly tighter than Onset because Foote already operates on a
smoothed novelty curve and produces fewer spurious peaks.

**Agglomerative — `n_segments: 8`**
Hard target of 8 segments — a reasonable approximation of typical pop song
structure (intro, verse ×2, chorus ×2, bridge, outro, etc.). Raise to 10–12 for
longer or more complex tracks. The algorithm will produce at most `n_segments`
sections regardless of track length.

**Laplacian — `k: 8`, `bandwidth: 0.5`, `min_seg_duration: 10.0 s`**
`k` is the number of Laplacian eigenvectors used (and the k-means target). The
default was raised from 6 → 8 to allow finer structure to emerge.

`bandwidth` controls the recurrence matrix: lower values mean only very similar
frames are connected, producing tighter clusters. 0.5 is a good middle ground;
lower it (e.g. 0.3) for more distinct sections, raise it (e.g. 0.8) if the graph
becomes too sparse.

`min_seg_duration` was added (and set high at 10 s) because the raw Laplacian
output marks a boundary at *every beat where the cluster label changes*, which
produces dozens of tiny 1–2 s fragments when cluster assignments alternate
quickly. The 10 s floor merges these into musically meaningful sections. Reduce
to 5–6 s if you want finer granularity.

**Spectral Clustering — `n_segments: 8`**
Same reasoning as Agglomerative. Spectral clustering operates on a frame-level
affinity matrix (downsampled to ≤300 frames for speed), so the actual number of
boundaries may be slightly fewer than `n_segments` after label smoothing.

### Quick-reference table

| Param | Algorithm | Effect |
|---|---|---|
| `min_seg_duration` | Onset, Foote, Laplacian | Min seconds between boundaries — raise for fewer, coarser segments |
| `kernel_size` | Foote | Larger = more temporal context per boundary decision |
| `k` | Laplacian | Target cluster/section count; also controls eigenvector depth |
| `bandwidth` | Laplacian | Recurrence graph connectivity — lower = stricter similarity |
| `n_segments` | Agglomerative, Spectral | Target section count |
| `tiny_beats` | Hybrid | Max length (in **beats**) for a section to count as a transition; tempo-relative (`32 * 60 / bpm`) |

---

## Verifying the Hybrid — `test_hybrid.py`

`test_hybrid.py` is a small stand-alone checker for the **Hybrid (Allin1+Agglom)**
algorithm. The Hybrid's "tiny section" cutoff is **tempo-relative**: a section
counts as a transition when it is at most **32 beats** long
(`tiny_thresh = 32 * 60 / bpm`, ≈ 8 bars in 4/4), rather than a fixed number of
seconds. This script shows, per song, how that tempo-aware threshold changes the
Hybrid boundaries versus the old fixed-8 s rule.

It reuses the **cached** `Allin1 (SOTA)` and `Agglomerative` boundaries already in
`results.json`, so it does **not** need Allin1 (or its brittle natten/torch stack)
to run — only librosa and the audio files. This is the recommended way to iterate
on the Hybrid without a working Allin1 install.

### Run it

```bash
source segmentation_env_311/bin/activate     # or your own venv
python test_hybrid.py            # print-only: BPM + old vs new boundaries
python test_hybrid.py --write    # also patch the new Hybrid rows into results.json
```

For each song it prints the librosa-detected BPM, the old (fixed 8 s) threshold vs
the new (32-beat) threshold, both boundary lists, and whether the Hybrid changed.

### See it in the dashboard

```bash
python test_hybrid.py --write    # required — writes the new Hybrid to results.json
python -m http.server 8080
# open http://localhost:8080/viewer.html
```

Without `--write` the script computes the new Hybrid but discards it after
printing, so the dashboard would look unchanged.

> **Requirements:** `results.json` must already contain `Allin1 (SOTA)` and
> `Agglomerative` rows for each song (they ship pre-computed). Tempo is detected
> from the files in `audio/`, so those must be present (`python segment.py`
> downloads them on first run).

> **Tuning:** the beat count lives in `seg_hybrid(..., tiny_beats=32)` in
> `segment.py`. Raise it (e.g. 33–34) to treat slightly longer sections as
> transitions; lower it to be stricter. BPM is estimated with librosa and may
> occasionally be an octave off (½× or 2×) — sanity-check the printed value.

---

## Dashboard tabs

- **🎧 Interactive Playback** — WaveSurfer waveform (click anywhere to seek).
  A red playhead sweeps across a stacked segment row per algorithm in real time.
  A live readout shows which segment each algorithm is currently in.
- **📊 Segment Timeline** — Plotly gantt chart with waveform overlay; zoomable and hoverable.
- **📋 Metrics** — Table: #segments, avg/min/max/std duration, exact boundary timestamps.
