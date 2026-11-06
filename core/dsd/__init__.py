"""DSD core: measurements, aesthetic rules, geometric design, validation and lab export.

Pure Python (no image libraries), ported from the tested JavaScript app and checked against it
for identical results (tests/test_parity.py). This is the server-side heart of the system.
"""
from .constants import TEETH, DEFAULT_RULES, PROPORTIONS, SHADES, TOOTH_NAME
from .measure import Scale, measure, scale_from_caliper, scale_from_iris
from .face import face_analysis, rest_analysis, iris_mm_per_px, face_points_from_landmarks
from .rules import evaluate_rules
from .design import (propose_design, apply_smile_arc, limit_changes, design_from_face, fill_missing,
                     describe_change, tooth_polygon, default_teeth, chain_y, papilla_points)
from .validate import validate_design, safety_checks, LIMITS
from .export import design_json, design_svg

__version__ = '0.1.0'
