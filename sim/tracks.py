"""Real-world-inspired circuits.

Each layout is a hand-made, simplified outline of the real circuit: the same
kind of corners in the same order, with the right proportions, but NOT survey
accurate. Control points are in arbitrary units; a smooth spline is fitted
through them and scaled to (real length x GAME_SCALE).
Control point 0 always sits on the start/finish straight.
"""
import numpy as np

from sim.track import Track

GAME_SCALE = 0.5      # game lap length = real lap length * GAME_SCALE
SPACING = 4.0         # meters between centerline points

CIRCUITS = {
    "Speedway Oval": dict(
        country="USA", km=4.0, width=16.0,
        note="Four fast corners. The easiest track: a good first test for the AI.",
        points=[(0, -4), (6, -4), (9.5, -2.8), (10.5, 0), (9.5, 2.8), (6, 4), (0, 4),
                (-6, 4), (-9.5, 2.8), (-10.5, 0), (-9.5, -2.8), (-6, -4)],
    ),
    "Monza": dict(
        country="Italy", km=5.79, width=13.0,
        note="Long straights, chicanes and a big final corner. All about braking late.",
        points=[(0, 26), (0, 18), (2.0, 15.0), (-0.3, 12.6), (1.8, 10.2),     # pit straight, 1st chicane
                (3.8, 6.8), (7.5, 4.4), (11.5, 3.8),                           # Curva Grande
                (14.0, 3.2), (15.4, 1.8), (17.0, 3.2),                         # 2nd chicane
                (19.0, 5.0), (19.8, 8.0), (19.2, 10.6),                        # the two Lesmo bends
                (16.5, 16.5), (13.5, 22.0),                                    # back straight
                (12.4, 24.3), (13.8, 25.8), (12.0, 27.4),                      # Ascari chicane
                (8.0, 30.8), (3.0, 31.0), (0.0, 28.8)],                        # final big corner
    ),
    "Silverstone": dict(
        country="UK", km=5.89, width=13.0,
        note="Fast sweeping corners and flowing esses, with a few slow bends in between.",
        points=[(0, 22), (7.0, 22.4), (11.0, 21.0), (13.2, 18.6), (12.0, 16.2),   # straight, fast right-left
                (13.4, 13.6), (16.2, 12.2), (17.6, 9.8), (15.8, 7.6), (12.6, 8.0), # tight right-hander, loop
                (10.0, 6.0), (9.2, 2.6), (6.0, 0.8), (2.0, 1.4),                   # flowing section, long straight
                (-1.0, 3.8), (-1.8, 7.0), (-0.2, 9.4), (1.6, 11.8), (0.0, 14.2),   # esses
                (-1.6, 17.0), (-1.4, 20.0)],                                       # last long right-hander
    ),
    "Spa-Francorchamps": dict(
        country="Belgium", km=7.0, width=13.0,
        note="Longest lap: a hairpin, a steep kink, a huge straight and fast sweepers.",
        points=[(0, 30), (-2.5, 31.2), (-5.0, 30.0), (-5.0, 27.4), (-2.6, 26.2),   # La Source hairpin
                (-0.5, 23.6), (1.8, 21.6), (1.2, 19.2), (3.4, 17.4),               # downhill kink
                (6.0, 12.0), (8.5, 6.0), (10.6, 1.4),                              # long straight
                (12.6, 0.6), (14.0, 2.4), (16.0, 3.0),                             # Les Combes chicane
                (17.5, 6.0), (16.0, 9.6), (13.0, 11.0), (12.4, 14.4),              # Malmedy, tight right
                (15.0, 17.6), (18.0, 20.6), (19.0, 24.4),                          # fast double-left
                (17.6, 28.4), (14.0, 30.6), (10.0, 31.4), (6.0, 32.0), (3.0, 31.4)],  # fast left to the bus stop
    ),
    "Interlagos": dict(
        country="Brazil", km=4.31, width=13.0,
        note="Runs anticlockwise. A tight, twisty lap with a steep run back to the start.",
        points=[(0, 18), (0, 12), (-1.4, 9.0), (0.6, 6.6), (3.6, 7.6), (5.0, 4.4),   # pit straight, S bends
                (8.0, 2.4), (12.0, 3.0), (14.6, 5.4),                                 # sweeping left, back straight
                (15.4, 9.0), (13.2, 11.6), (10.4, 11.2), (8.2, 13.4),                 # right-hander, tight infield
                (9.2, 16.4), (12.0, 17.8), (13.0, 20.6), (10.6, 23.0),               # hairpin-style bends
                (6.6, 22.6), (3.8, 20.6)],                                            # climb back to the line
    ),
    "Hairpin Haven": dict(
        country="Training Grounds", km=2.8, width=11.0,
        note="Alternating 180-degree hairpins and heavy braking zones. Teaches threshold braking.",
        points=[
            (0.0, 0.0), (0.0, 10.0), (0.0, 20.0), (2.5, 22.0), (5.0, 20.0), (5.0, 10.0),
            (5.0, 2.0), (7.5, 0.0), (10.0, 2.0), (10.0, 10.0), (10.0, 18.0), (12.5, 20.0),
            (15.0, 18.0), (15.0, 8.0), (15.0, -2.0), (10.0, -5.0), (3.0, -4.0),
        ],
    ),
    "Chicane Alley": dict(
        country="Training Grounds", km=3.2, width=10.5,
        note="High speed straights intersected by sharp chicane complexes. Teaches quick steering flicking.",
        points=[
            (0.0, 0.0), (0.0, 12.0), (-2.5, 16.0), (2.5, 20.0), (0.0, 24.0), (0.0, 36.0),
            (4.0, 40.0), (10.0, 42.0), (16.0, 40.0), (18.0, 34.0), (20.5, 28.0), (15.5, 24.0),
            (18.0, 20.0), (18.0, 8.0), (14.0, 0.0), (10.0, -3.0), (6.0, 2.0), (2.5, -1.0),
        ],
    ),
    "Technical Karting Ring": dict(
        country="Training Grounds", km=2.4, width=9.5,
        note="Compact, narrow twisting stadium track with continuous 90 and 120 degree turns.",
        points=[
            (0.0, -6.0),   # S/F straight bottom
            (0.0, 0.0),    # S/F line
            (0.0, 10.0),   # S/F straight top
            (4.0, 14.0),   # T1 sharp right
            (10.0, 14.0),  # burst
            (13.0, 10.0),  # T2 sharp right
            (13.0, 4.0),   # burst
            (17.0, 1.0),   # T3 sharp left
            (22.0, 4.0),   # burst
            (24.0, 9.0),   # T4 hairpin right
            (21.0, 14.0),  # hairpin exit
            (16.0, 17.0),  # T5 left
            (10.0, 19.0),  # S-bend
            (4.0, 18.0),
            (-1.0, 18.0),  # T6 left
            (-6.0, 13.0),  # T7 sharp left
            (-6.0, 4.0),   # drop
            (-4.0, -3.0),  # T8 final sweeper
            (-1.5, -6.0),  # connect into S/F straight
        ],
    ),
    "Snake Esses": dict(
        country="Training Grounds", km=3.0, width=11.0,
        note="Fast rhythmic undulating esses and decreasing radius turns. Teaches car balance.",
        points=[
            (0.0, 0.0), (0.0, 10.0), (3.5, 14.0), (-3.5, 19.0), (4.0, 24.0), (-4.5, 29.0),
            (0.0, 35.0), (6.0, 38.0), (14.0, 36.0), (18.0, 30.0), (16.0, 22.0), (12.0, 16.0),
            (14.0, 9.0), (12.0, 2.0), (7.0, -2.0), (2.5, -2.0),
        ],
    ),
    "Monza-Spa Hyper Circuit": dict(
        country="Italy/Belgium", km=11.6, width=13.0,
        note="True hybrid fusion of Monza and Spa-Francorchamps (~1.45 min lap, 5.8km). 330+ km/h straights crashing into the Rettifilo chicane, Curva Grande, Roggia, Eau Rouge / Raidillon uphill compression, Kemmel blast, Les Combes chicane, Blanchimont, and Bus Stop into the sweeping Parabolica.",
        points=[
            (0.0, 0.0),      # S/F straight (Monza pit straight)
            (0.0, 20.0),     # Main straight blast (330 km/h)
            (1.5, 23.5),     # Rettifilo Chicane entry (hard threshold brake)
            (-1.5, 25.0),    # Rettifilo Chicane flick
            (0.5, 27.5),     # Rettifilo Chicane exit
            (6.0, 31.0),     # Curva Grande sweeper start
            (13.0, 31.5),    # Curva Grande sweeper mid
            (17.5, 28.0),    # Curva Grande sweeper exit
            (19.0, 24.5),    # Roggia approach
            (21.0, 22.5),    # Roggia chicane right
            (19.0, 20.5),    # Roggia chicane left exit
            (18.5, 15.0),    # Transition towards Spa sector
            (17.0, 11.0),    # Eau Rouge compression left
            (19.0, 7.5),     # Raidillon uphill crest right
            (18.0, 4.0),     # Raidillon exit
            (15.0, -4.0),    # Kemmel Straight (330 km/h blast)
            (12.0, -12.0),   # Kemmel Straight continue
            (8.5, -19.0),    # Kemmel Straight end
            (6.0, -21.5),    # Les Combes Chicane entry (heavy brake)
            (7.5, -24.0),    # Les Combes chicane flick
            (5.0, -26.5),    # Les Combes exit / Malmedy entry
            (1.0, -28.0),    # Malmedy sweeper
            (-4.0, -26.0),   # Bruxelles double-left descent
            (-8.0, -22.0),   # Bruxelles exit
            (-12.0, -16.0),  # Blanchimont ultra-high-speed sweeper
            (-14.0, -9.0),   # Blanchimont exit
            (-14.5, -4.0),   # Bus Stop chicane approach
            (-12.5, -2.0),   # Bus Stop chicane right (hard brake)
            (-14.5, 0.0),    # Bus Stop chicane left exit
            (-11.0, 2.5),    # Curva Parabolica entry
            (-6.0, 1.5),     # Curva Parabolica sweeper mid
            (-2.0, -0.5),    # Curva Parabolica exit into S/F straight
        ],
    ),
    "Monte-Suzuka Grand Ring": dict(
        country="Monaco/Japan", km=11.5, width=12.0,
        note="True hybrid fusion of Monaco and Suzuka (~1.45 min lap, 5.75km). Sainte-Dévote 90° braking, Beau Rivage climb, Casino Square, Suzuka's high-g rhythmic S-Curves, Dunlop sweeper, Degners 1 & 2, Monaco Grand Hotel Hairpin, 320 km/h Harbor Tunnel, Suzuka 130R, and Casio Triangle chicane.",
        points=[
            (0.0, 0.0),      # S/F straight (Monaco boulevard)
            (0.0, 16.0),     # Main straight blast
            (2.5, 20.0),     # Sainte-Dévote 90-deg right (hard brake)
            (3.5, 25.0),     # Beau Rivage uphill climb
            (5.5, 29.0),     # Massenet fast left
            (8.5, 30.5),     # Casino Square right bend
            (11.5, 28.5),    # Mirabeau approach
            (14.5, 25.0),    # Suzuka S-Curves entry (T1 right)
            (13.0, 20.5),    # S-Curves T2 (left flick)
            (16.0, 17.0),    # S-Curves T3 (right flick)
            (14.5, 12.5),    # Dunlop Curve (sweeping left uphill)
            (16.5, 8.0),     # Degner 1 fast right
            (17.5, 4.5),     # Degner 2 tight 90-deg right (heavy brake)
            (14.5, 1.0),     # Monaco Hairpin approach descent
            (11.5, -0.5),    # Grand Hotel Hairpin (slowest hairpin in racing, ultra-tight)
            (9.0, 1.0),      # Portier right corner onto coast
            (7.0, -3.0),     # Tunnel entry
            (5.0, -9.0),     # Monaco Harbor Tunnel (flat out 320 km/h)
            (3.0, -16.0),    # Tunnel blast continue
            (0.0, -21.0),    # Suzuka 130R approach
            (-4.0, -23.0),   # 130R flat-out sweeping left (280 km/h)
            (-9.0, -21.0),   # 130R exit into Casio Triangle braking zone
            (-11.5, -16.5),  # Casio Triangle Chicane right (hard threshold brake)
            (-10.0, -12.5),  # Casio Triangle Chicane left exit
            (-7.5, -7.5),    # Final bend
            (-3.0, -2.5),    # Return onto S/F straight
        ],
    ),
    "Interlagos-Silverstone Colosseum": dict(
        country="Brazil/UK", km=11.8, width=13.0,
        note="True hybrid fusion of Interlagos and Silverstone (~1.5 min lap, 5.9km). Downhill Senna 'S' into Curva do Sol, Wellington Straight blast, Silverstone's aerodynamic Maggotts-Becketts-Chapel esses, Hangar Straight, Stowe braking zone, Luffield carousel, Pinheiro hairpin, and Junção climb.",
        points=[
            (0.0, 0.0),       # S/F straight mid
            (0.0, 10.0),      # S/F line and acceleration
            (0.0, 21.0),      # Main straight blast
            (-3.0, 25.5),     # Senna 'S' Turn 1 downhill left (heavy brake)
            (-1.5, 29.0),     # Senna 'S' Turn 2 uphill right
            (3.0, 30.5),      # Senna 'S' exit into Curva do Sol
            (8.0, 31.0),      # Curva do Sol sweeping fast left
            (14.0, 31.5),     # Wellington straight entry
            (22.0, 31.5),     # Wellington straight blast (330 km/h)
            (29.0, 30.0),     # Maggotts entry
            (31.5, 25.5),     # Maggotts high-speed flick
            (29.5, 20.5),     # Becketts sharp left
            (32.0, 15.5),     # Becketts sharp right
            (30.5, 10.0),     # Chapel sweeper exit
            (29.0, 3.0),      # Hangar Straight blast (335 km/h)
            (27.0, -5.0),     # Hangar Straight continue
            (25.0, -13.0),    # Stowe braking zone
            (21.5, -17.5),    # Stowe tight right-hander (heavy brake)
            (16.0, -18.5),    # Vale burst
            (11.5, -20.5),    # Luffield entry
            (7.0, -19.5),     # Luffield carousel mid (sweeping loop)
            (2.5, -16.0),     # Luffield exit / infield blast
            (-2.0, -14.0),    # Pinheiro approach
            (-6.5, -10.5),    # Pinheiro hairpin entry (threshold brake)
            (-8.0, -5.5),     # Pinheiro hairpin apex
            (-5.5, -2.0),     # Junção left climb
            (-2.5, -0.5),     # Final acceleration onto start/finish straight
        ],
    ),
}


# ---------------------------------------------------------------------------
# TEST CIRCUITS — held-out evaluation set. NEVER add these to CURRICULUM.
# These 5 circuits measure how well the trained AI generalises to unseen tracks.
# ---------------------------------------------------------------------------
TEST_CIRCUITS = {
    "Suzuka": dict(
        country="Japan", km=5.807, width=12.0,
        note="S-curves, Degner, a hairpin, Spoon, fast 130R and a chicane. Very technical.",
        points=[
            (0, 26),      # Start/finish
            (0, 16),      # End of main straight
            (4.0, 12.5),  # Turn 1 (right)
            (7.0, 11.5),  # S-curves T2 (left)
            (8.5, 14.0),  # S-curves T3 (right)
            (12.0, 13.0), # Dunlop sweep (right)
            (15.0, 10.0), # Degner 1 (right)
            (17.0, 6.5),  # Degner 2
            (15.5, 3.0),  # Hairpin approach
            (12.0, 1.5),  # Hairpin left (wide)
            (8.5, 3.0),   # Hairpin exit
            (5.5, 7.5),   # Spoon / 130R
            (-2.0, 16.5), # 130R exit
            (-3.5, 23.0), # Chicane exit, back to line
        ],
    ),
    "Monaco": dict(
        country="Monaco", km=3.337, width=9.0,
        note="The narrowest street circuit. Grand Hotel hairpin, tunnel, swimming pool esses.",
        points=[
            (0, 14),     # Start/finish straight
            (0, 10),     # Straight end
            (1.5, 7.5),  # Sainte Devote (right)
            (2.5, 5.5),  # Beau Rivage climb
            (4.5, 4.5),  # Massenet (right)
            (6.5, 4.0),  # Casino Square (right)
            (8.0, 5.0),  # Mirabeau (right)
            (8.8, 7.0),  # Grand Hotel hairpin approach
            (8.0, 9.0),  # Grand Hotel hairpin (tight right)
            (6.0, 9.5),  # Portier exit (right)
            (5.0, 8.5),  # Into tunnel (going right)
            (3.5, 9.5),  # Tunnel
            (2.5, 11.0), # Swimming pool S1 (left)
            (3.5, 12.5), # Swimming pool S2 (right)
            (2.0, 13.5), # La Rascasse (right)
            (0.5, 13.8), # Anthony Noghes
        ],
    ),
    "Bahrain": dict(
        country="Bahrain", km=5.412, width=12.0,
        note="Long main straight, heavy braking T1, fast middle sector, tight stadium section.",
        points=[
            (0.0, 0.0),    # Start/finish straight mid
            (0.0, 8.0),    # S/F straight top
            (0.0, 14.0),   # Heavy braking T1 entry
            (2.5, 12.0),   # T1-T2 hairpin right
            (4.0, 10.0),   # T3 exit
            (6.0, 9.5),    # T4 right
            (8.5, 7.5),    # T5-T6 downhill sweep
            (8.0, 5.0),    # T7
            (6.5, 4.0),    # T8 hairpin
            (4.5, 4.5),    # T9 approach
            (3.5, 3.0),    # T10 sharp left
            (3.5, 0.5),    # Back straight
            (5.0, -1.5),   # T11-T12 sweep
            (4.5, -3.5),   # T13 right
            (2.5, -3.0),   # T14 right
            (0.8, -1.5),   # T15 final turn onto straight
        ],
    ),
    "Albert Park": dict(
        country="Australia", km=5.303, width=12.0,
        note="Melbourne street circuit. Mix of fast sweepers and tight chicanes around a lake.",
        points=[
            (0.0, 0.0),    # Start/finish line mid
            (0.0, 8.0),    # Main straight
            (0.0, 15.0),   # Heavy braking T1
            (2.5, 17.5),   # T1 (right)
            (5.0, 16.5),   # T2 (left)
            (7.5, 13.0),   # T3 (sharp right)
            (9.0, 10.0),   # T4 (left)
            (11.0, 11.0),  # T5 (right sweep)
            (13.0, 8.5),   # T6 (right)
            (13.5, 5.0),   # T7 (right)
            (11.5, 2.0),   # T8 (left sweep around lake)
            (8.5, 0.0),    # T9 (fast chicane approach)
            (5.5, -3.0),   # T10 (fast chicane right)
            (3.5, -6.0),   # T11-T12 (lake esses)
            (1.5, -10.0),  # T13 (right)
            (-2.0, -11.0), # T14 (hairpin right)
            (-2.5, -7.0),  # T15 (left chicane approach)
            (-1.5, -4.0),  # T16 (chicane onto straight)
            (0.0, -3.0),   # S/F straight bottom
        ],
    ),
    "Red Bull Ring": dict(
        country="Austria", km=4.318, width=11.0,
        note="Short, aggressive Alpine circuit. Heavy braking into T3 and fast sweepers.",
        points=[
            (0.0, 0.0),     # S/F straight mid
            (0.0, 6.0),     # S/F straight top
            (2.0, 7.5),     # T1 (Niki Lauda Kurve, right)
            (6.0, 9.5),     # Schönberg straight uphill
            (11.0, 11.5),   # Straight apex
            (13.0, 10.5),   # T3 (tight right hairpin)
            (9.5, 6.5),     # Downhill run to T4
            (7.0, 3.5),     # T4 (right)
            (5.5, 1.5),     # T5-T6 (fast left/right sweep)
            (4.5, -1.0),    # T7 (right)
            (3.5, -2.5),    # T8 (left/right transition)
            (2.0, -3.0),    # T9 (Jochen Rindt Kurve, right)
            (0.8, -1.5),    # T10 (final right back onto main straight)
        ],
    ),
}


def _spline(points, per_seg=24, alpha=0.5):
    """Closed centripetal Catmull-Rom spline through the points (no overshoot)."""
    P = np.asarray(points, dtype=float)
    n = len(P)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** alpha
        t2 = t1 + np.linalg.norm(p2 - p1) ** alpha
        t3 = t2 + np.linalg.norm(p3 - p2) ** alpha
        t = np.linspace(t1, t2, per_seg, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.vstack(out)


def _closed_length(pts):
    return np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1).sum()


def _resample(pts, spacing):
    """Evenly spaced points along a closed polyline."""
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    n = max(int(round(cum[-1] / spacing)), 8)
    s = np.linspace(0, cum[-1], n, endpoint=False)
    return np.stack([np.interp(s, cum, closed[:, 0]), np.interp(s, cum, closed[:, 1])], axis=1)


def circuit_names():
    return list(CIRCUITS)


def load_circuit(name, game_scale=GAME_SCALE):
    spec = CIRCUITS[name]
    dense = _spline(spec["points"])
    factor = spec["km"] * 1000.0 * game_scale / _closed_length(dense)
    center = _resample(dense * factor, SPACING)
    track = Track(center, spec["width"], name=name)
    track.country, track.note, track.real_km = spec["country"], spec["note"], spec["km"]
    return track


def test_circuit_names():
    """Names of the held-out evaluation circuits (never used in training)."""
    return list(TEST_CIRCUITS)


def load_test_circuit(name, game_scale=GAME_SCALE):
    """Load one of the held-out test circuits. Do NOT use in train_all.py."""
    spec = TEST_CIRCUITS[name]
    dense = _spline(spec["points"])
    factor = spec["km"] * 1000.0 * game_scale / _closed_length(dense)
    center = _resample(dense * factor, SPACING)
    track = Track(center, spec["width"], name=name)
    track.country, track.note, track.real_km = spec["country"], spec["note"], spec["km"]
    return track


# ---------------- validation ----------------
def min_corner_radius(track):
    """Tightest centerline corner radius in meters."""
    t = track.tangent
    ang = np.abs(np.arctan2(t[:, 0] * np.roll(t, -1, 0)[:, 1] - t[:, 1] * np.roll(t, -1, 0)[:, 0],
                            (t * np.roll(t, -1, 0)).sum(axis=1)))
    step = np.linalg.norm(np.roll(track.center, -1, 0) - track.center, axis=1)
    return float((step / np.maximum(ang, 1e-9)).min())


def _crossings(segs):
    """Number of crossing pairs among non-neighbouring segments of one polyline set."""
    a, b = segs[:, 0], segs[:, 1]
    m, count = len(segs), 0

    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])

    for i in range(m):
        j = np.arange(i + 2, m)
        j = j[~((i == 0) & (j == m - 1))]
        if len(j) == 0:
            continue
        o1, o2 = orient(a[i], b[i], a[j]), orient(a[i], b[i], b[j])
        o3, o4 = orient(a[j], b[j], a[i]), orient(a[j], b[j], b[i])
        count += int(np.sum((o1 * o2 < 0) & (o3 * o4 < 0)))
    return count


def validate(track):
    """Problems found with a track (empty list = fine)."""
    problems = []
    r = min_corner_radius(track)
    if r < track.width / 2 * 1.05:
        problems.append(f"corner radius {r:.1f} m is too tight for width {track.width} m")
    n = len(track.left)
    bad = _crossings(np.stack([track.left, np.roll(track.left, -1, 0)], axis=1)) \
        + _crossings(np.stack([track.right, np.roll(track.right, -1, 0)], axis=1))
    if bad:
        problems.append(f"track edges cross themselves {bad} times")
    # track must not run too close to another part of itself
    k = int(8 * track.width / SPACING)
    d = np.linalg.norm(track.center[:, None] - track.center[None], axis=2)
    idx = np.arange(n)
    gap = np.minimum(np.abs(idx[:, None] - idx[None]), n - np.abs(idx[:, None] - idx[None]))
    d[gap < k] = np.inf
    if d.min() < track.width * 1.5:
        problems.append(f"track passes within {d.min():.1f} m of itself")
    return problems
