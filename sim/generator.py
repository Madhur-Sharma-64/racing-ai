"""Procedural Track Generator based on Hand-Drawn Archetypes.

Transforms the user's sketch archetypes (The Cross, Amoeba, The Great Turn,
Big Mountain, Bacteria, Teardrop, Giant Whale, Glyph, Peanut) into
infinite procedural racing circuits with physically validated splines.
"""
from enum import Enum
import numpy as np

from sim.track import Track
from sim.tracks import _spline, _resample, validate, SPACING, GAME_SCALE


class Archetype(str, Enum):
    CROSS = "The Cross"
    AMOEBA = "The Amoeba"
    GREAT_TURN = "The Great Turn"
    MOUNTAIN = "Big Mountain"
    TEARDROP = "The Teardrop"
    BACTERIA = "Bacteria"
    WHALE = "Giant Whale"
    GLYPH = "Complex Glyph"
    PEANUT = "Kidney Peanut"


# Master control point templates derived from hand sketches
ARCHETYPE_TEMPLATES = {
    Archetype.CROSS: [
        (2.0, 2.0), (8.0, 2.0), (10.0, 0.0), (8.0, -2.0), (2.0, -2.0),
        (2.0, -8.0), (0.0, -10.0), (-2.0, -8.0), (-2.0, -2.0),
        (-8.0, -2.0), (-10.0, 0.0), (-8.0, 2.0), (-2.0, 2.0),
        (-2.0, 8.0), (0.0, 10.0), (2.0, 8.0)
    ],
    Archetype.AMOEBA: [
        # 5 distinct radiating pseudopod lobes
        (13.5, 0.0), (10.0, 5.0), (6.5, 9.0), (2.0, 12.0), (-3.0, 10.0),
        (-8.0, 11.5), (-11.0, 7.0), (-8.5, 2.0), (-12.0, -2.5), (-11.5, -8.0),
        (-6.0, -9.0), (-2.0, -12.5), (3.0, -10.0), (8.5, -11.0), (10.5, -5.5)
    ],
    Archetype.GREAT_TURN: [
        (0.0, -10.0), (0.0, 0.0), (0.0, 12.0), (0.0, 24.0),
        (3.0, 27.0), (6.0, 27.0), (9.0, 24.0),
        (9.0, 12.0), (9.0, 0.0),
        (12.0, -4.0), (14.0, -8.0), (12.0, -12.0), (8.0, -12.0),
        (5.0, -8.0), (2.0, -10.0)
    ],
    Archetype.MOUNTAIN: [
        (-8.0, -4.0), (0.0, -4.0), (8.0, -4.0),
        (10.0, -1.0), (9.0, 4.0), (6.0, 9.0), (4.0, 7.0),
        (2.0, 3.0), (0.0, 1.0), (-2.0, 3.0), (-4.0, 7.0),
        (-6.0, 9.0), (-9.0, 4.0), (-10.0, -1.0)
    ],
    Archetype.TEARDROP: [
        (0.0, -6.0), (6.0, -6.0), (12.0, 0.0), (6.0, 6.0), (0.0, 6.0),
        (-6.0, 4.0), (-12.0, 2.0), (-16.0, 0.0), (-12.0, -2.0), (-6.0, -4.0)
    ],
    Archetype.BACTERIA: [
        (0.0, -5.0), (4.0, -4.5), (7.0, -6.0), (10.0, -4.5), (13.0, -6.0), (15.0, -3.0), (16.0, 0.0),
        (15.0, 3.0), (13.0, 6.0), (10.0, 4.5), (7.0, 6.0), (4.0, 4.5), (0.0, 5.0),
        (-4.0, 4.5), (-7.0, 6.0), (-10.0, 4.5), (-13.0, 6.0), (-15.0, 3.0), (-16.0, 0.0),
        (-15.0, -3.0), (-13.0, -6.0), (-10.0, -4.5), (-7.0, -6.0), (-4.0, -4.5)
    ],
    Archetype.WHALE: [
        (0.0, -4.0), (8.0, -4.0), (14.0, -2.0), (18.0, 2.0), (16.0, 6.0), (10.0, 7.0), (4.0, 5.0),
        (0.0, 3.0), (-4.0, 1.0), (-7.0, 4.0), (-11.0, 6.0), (-14.0, 4.0), (-12.0, 0.0),
        (-14.0, -4.0), (-11.0, -6.0), (-7.0, -4.0), (-4.0, -1.0)
    ],
    Archetype.GLYPH: [
        (0.0, -6.0), (6.0, -6.0), (10.0, -3.0), (12.0, 2.0), (16.0, 5.0), (12.0, 6.0), (8.0, 6.0),
        (6.0, 10.0), (0.0, 10.0), (-6.0, 8.0), (-8.0, 4.0), (-10.0, 0.0), (-8.0, -4.0), (-4.0, -6.0)
    ],
    Archetype.PEANUT: [
        (0.0, -5.0), (6.0, -5.0), (10.0, -2.0), (11.0, 2.0), (8.0, 5.0), (3.0, 3.5),
        (-2.0, 2.0), (-6.0, 5.0), (-10.0, 4.0), (-12.0, 0.0), (-9.0, -4.0), (-4.0, -4.5)
    ],
}


def get_archetype_names():
    """Return list of all available drawing archetype names."""
    return [a.value for a in Archetype]


def generate_track(
    archetype=None,
    seed=None,
    km=3.0,
    width=12.0,
    jitter_strength=0.12,
    max_attempts=30,
    game_scale=GAME_SCALE,
):
    """Generate a valid, playable procedural Track instance based on sketch archetypes.

    Parameters:
        archetype: Archetype enum or string name. If None, chosen randomly.
        seed: Integer random seed for reproducible procedural generation.
        km: Target real-world circuit length in kilometers.
        width: Track width in meters.
        jitter_strength: Magnitude of coordinate perturbation (0.0 = exact sketch, 0.25 = extreme).
        max_attempts: Validation retry attempts with progressive jitter damping.
        game_scale: Game physics scale factor.

    Returns:
        Track object with valid centerline, borders, normals, and metadata.
    """
    rng = np.random.default_rng(seed)

    if archetype is None:
        arch_choice = rng.choice(list(Archetype))
    elif isinstance(archetype, str):
        # Match string to Archetype enum
        found = False
        for a in Archetype:
            if a.value.lower() == archetype.lower() or a.name.lower() == archetype.lower():
                arch_choice = a
                found = True
                break
        if not found:
            arch_choice = Archetype.CROSS
    else:
        arch_choice = archetype

    base_pts = np.asarray(ARCHETYPE_TEMPLATES[arch_choice], dtype=float)

    for attempt in range(max_attempts):
        damping = 1.0 / (1.0 + attempt * 0.15)
        curr_jitter = jitter_strength * damping

        # 1. Random aspect ratio scaling (elongation/squashing)
        sx = rng.uniform(0.85, 1.15)
        sy = rng.uniform(0.85, 1.15)
        pts = base_pts.copy()
        pts[:, 0] *= sx
        pts[:, 1] *= sy

        # 2. Random rotation
        angle = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)
        rot_mat = np.array([[c, -s], [s, c]])
        pts = pts @ rot_mat.T

        # 3. Local coordinate jittering
        if curr_jitter > 0.0:
            segs = np.roll(pts, -1, axis=0) - pts
            seg_lens = np.linalg.norm(segs, axis=1, keepdims=True)
            noise = rng.normal(0.0, curr_jitter * seg_lens, size=pts.shape)
            # Anchor the starting point (index 0) to keep start/finish straight smooth
            noise[0] = 0.0
            pts += noise

        # 4. Generate spline and scale
        dense = _spline(pts)
        closed_len = np.linalg.norm(np.roll(dense, -1, axis=0) - dense, axis=1).sum()
        if closed_len < 1e-3:
            continue

        factor = km * 1000.0 * game_scale / closed_len
        center = _resample(dense * factor, SPACING)

        # Align start/finish line to the longest straight (lowest curvature)
        tangs = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
        t_angles = np.unwrap(np.arctan2(tangs[:, 1], tangs[:, 0]))
        curv = np.abs(np.gradient(t_angles))
        kernel = np.ones(15) / 15.0
        curv_smooth = np.convolve(curv, kernel, mode="same")
        best_start = int(np.argmin(curv_smooth))
        center = np.roll(center, -best_start, axis=0)

        track_name = f"{arch_choice.value} #{seed if seed is not None else int(rng.integers(1000, 9999))}"
        track = Track(center, width=width, name=track_name)

        # 5. Validation check
        problems = validate(track)
        if not problems:
            track.country = "Procedural"
            track.real_km = km
            track.note = f"Procedurally generated archetype: {arch_choice.value} (seed={seed})"
            track.archetype = arch_choice.value
            return track

    # Fallback to pristine un-jittered template if all jitter attempts failed
    dense = _spline(base_pts)
    factor = km * 1000.0 * game_scale / np.linalg.norm(np.roll(dense, -1, axis=0) - dense, axis=1).sum()
    center = _resample(dense * factor, SPACING)
    tangs = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    t_angles = np.unwrap(np.arctan2(tangs[:, 1], tangs[:, 0]))
    curv = np.abs(np.gradient(t_angles))
    kernel = np.ones(15) / 15.0
    curv_smooth = np.convolve(curv, kernel, mode="same")
    best_start = int(np.argmin(curv_smooth))
    center = np.roll(center, -best_start, axis=0)

    track = Track(center, width=width, name=f"{arch_choice.value} (Clean)")
    track.country = "Procedural"
    track.real_km = km
    track.note = f"Pristine sketch archetype: {arch_choice.value}"
    track.archetype = arch_choice.value
    return track


def generate_track_suite(n_tracks=50, seed_base=2000, km_range=(2.0, 4.5), width_range=(11.0, 13.5)):
    """Generate a diverse suite of validated procedural tracks across all archetypes."""
    suite = []
    archetypes = list(Archetype)
    for i in range(n_tracks):
        arch = archetypes[i % len(archetypes)]
        seed = seed_base + i
        rng = np.random.default_rng(seed)
        km = float(rng.uniform(km_range[0], km_range[1]))
        width = float(rng.uniform(width_range[0], width_range[1]))
        track = generate_track(archetype=arch, seed=seed, km=km, width=width)
        suite.append(track)
    return suite
