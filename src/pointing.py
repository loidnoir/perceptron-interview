from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as np

from shapely.geometry import Polygon as ShapelyPolygon
from skimage.measure import find_contours
from decimal import Decimal, ROUND_HALF_UP

def round_half_up(n):
    return int(Decimal(n).to_integral_value(rounding=ROUND_HALF_UP))


@dataclass
class NormalizationConfig:
    range_x: float
    range_y: float
    image_width: int
    image_height: int

    def to_prompt(self) -> str:
        assert self.range_x == self.range_y, "prompt assumes both x and y are normalized to the same range"
        return f"Points are specified as coordinates (x,y) where x and y range from 0 to {self.range_x}."


def is_valid_normalization_config(config_data: Optional[Any]) -> bool:
    """Checks if the provided data represents a valid NormalizationConfig."""
    if not isinstance(config_data, dict):
        return False
    try:
        NormalizationConfig(**config_data)
        return True
    except (TypeError, ValueError):  # Catches missing keys or invalid values
        return False
    except Exception:  # Catch unexpected errors during instantiation
        return False


NOT_A_POINT = "not_a_point"


@dataclass
class Point:
    """Base class for point types"""

    pass

    @classmethod
    @property
    def point_name(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def point_type(cls) -> str:
        raise NotImplementedError()

    @classmethod
    @property
    def prompt_definition(cls) -> str:
        raise NotImplementedError()

    def normalize(self, config: NormalizationConfig) -> "Point":
        raise NotImplementedError()

    def denormalize(self, config: NormalizationConfig) -> "Point":
        raise NotImplementedError()


@dataclass
class SinglePoint(Point):
    """A single point with x,y coordinates

    Example:
        point = SinglePoint(1, 2)
        str_repr = "(1,2)"
    """

    x: int
    y: int
    t: float | int | None = None
    mention: str | None = None

    @classmethod
    @property
    def point_name(cls) -> str:
        return "point"

    @classmethod
    def point_type(cls) -> str:
        return cls.point_name

    @classmethod
    @property
    def prompt_definition(cls) -> str:
        return "point defined as (x,y) coordinates, specified as <point> (x,y) </point>"

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def normalize(self, config: NormalizationConfig) -> "SinglePoint":
        norm_x = (self.x / config.image_width) * config.range_x
        norm_y = (self.y / config.image_height) * config.range_y
        return SinglePoint(x=round_half_up(norm_x), y=round_half_up(norm_y))

    def denormalize(self, config: NormalizationConfig) -> "SinglePoint":
        denorm_x = (self.x / config.range_x) * (config.image_width)
        denorm_y = (self.y / config.range_y) * (config.image_height)
        return SinglePoint(x=round_half_up(denorm_x), y=round_half_up(denorm_y))

    def serialize_coords(self) -> str:
        return f"({self.x},{self.y})"


@dataclass
class BoundingBox(Point):
    """A bounding box defined by top-left and bottom-right points

    Example:
        box = BoundingBox(SinglePoint(1,1), SinglePoint(5,5))
        str_repr = "(1,1),(5,5)"
    """

    top_left: SinglePoint
    bottom_right: SinglePoint
    t: float | int | None = None
    mention: str | None = None

    def __hash__(self) -> int:
        """
        Hash function for BoundingBox to enable use with memoization.

        Returns:
            int: A hash value based on the top_left and bottom_right points.
        """
        return hash((self.top_left, self.bottom_right))

    @classmethod
    @property
    def point_name(cls) -> str:
        return "point_box"

    @classmethod
    def point_type(cls) -> str:
        return "bbox"

    @classmethod
    @property
    def prompt_definition(cls) -> str:
        return "bounding box described as top left corner (x1, y1) and bottom right corner (x2, y2) presented as <point_box> (x1,y1) (x2,y2) </point_box>"

    def normalize(self, config: NormalizationConfig) -> "BoundingBox":
        """Normalizes both points of the bounding box"""
        return BoundingBox(
            top_left=self.top_left.normalize(config),
            bottom_right=self.bottom_right.normalize(config),
        )

    def denormalize(self, config: NormalizationConfig) -> "BoundingBox":
        return BoundingBox(
            top_left=self.top_left.denormalize(config),
            bottom_right=self.bottom_right.denormalize(config),
        )

    def serialize_coords(self) -> str:
        return f"({self.top_left.x},{self.top_left.y}) ({self.bottom_right.x},{self.bottom_right.y})"


@dataclass
class Polygon(Point):
    """A mask defined by a list of points forming a hull/polygon

    Example:
        mask = Polygon([SinglePoint(1,1), SinglePoint(2,2), SinglePoint(3,1)])
        str_repr = "(1,1),(2,2),(3,1)"
    """

    hull: List[SinglePoint]
    t: float | int | None = None
    mention: str | None = None

    @classmethod
    @property
    def point_name(cls) -> str:
        return "polygon"

    @classmethod
    def point_type(cls) -> str:
        return cls.point_name

    @classmethod
    @property
    def prompt_definition(cls) -> str:
        return "polygon described as a list of N (x,y) coordinates presented as <polygon> (x1,y1) (x2,y2) ... (xN, yN) </polygon>"

    def __hash__(self) -> int:
        return hash(tuple(self.hull))

    def normalize(self, config: NormalizationConfig) -> "Polygon":
        return Polygon(hull=[point.normalize(config) for point in self.hull])

    def denormalize(self, config: NormalizationConfig) -> "Polygon":
        return Polygon(hull=[point.denormalize(config) for point in self.hull])

    def serialize_coords(self) -> str:
        return " ".join(f"({pt.x},{pt.y})" for pt in self.hull)


def bbox_to_centroid(bbox: BoundingBox) -> SinglePoint:
    """
    Converts a BoundingBox into its centroid point.
    """
    # Calculate center x coordinate
    center_x = (bbox.top_left.x + bbox.bottom_right.x) // 2

    # Calculate center y coordinate
    center_y = (bbox.top_left.y + bbox.bottom_right.y) // 2

    return SinglePoint(center_x, center_y)


def polygon_to_centroid(polygon: Polygon) -> SinglePoint:
    """
    Calculate the centroid of a polygon defined by its hull points.
    Uses the formula for the centroid of a polygon:
    x = (1/6A) * sum((x[i] + x[i+1]) * (x[i] * y[i+1] - x[i+1] * y[i]))
    y = (1/6A) * sum((y[i] + y[i+1]) * (x[i] * y[i+1] - x[i+1] * y[i]))
    where A is the signed area of the polygon

    Args:
        polygon: Polygon containing hull points defining the polygon

    Returns:
        SinglePoint representing the centroid coordinates
    """
    # Convert hull points to x,y coordinate lists
    hull = polygon.hull
    if not hull:
        raise ValueError("Empty polygon hull")

    # Close the polygon by adding the first point at the end if needed
    if hull[0] != hull[-1]:
        hull = hull + [hull[0]]

    area = 0.0
    centroid_x = 0.0
    centroid_y = 0.0

    # Calculate area and weighted sums for centroid
    for i in range(len(hull) - 1):
        x0, y0 = hull[i].x, hull[i].y
        x1, y1 = hull[i + 1].x, hull[i + 1].y

        # Signed area of the triangle formed with the origin
        cross_term = (x0 * y1) - (x1 * y0)
        area += cross_term

        # Weighted sums
        centroid_x += (x0 + x1) * cross_term
        centroid_y += (y0 + y1) * cross_term

    # Complete the area calculation
    area = area / 2.0

    # Avoid division by zero
    if abs(area) < 1e-8:
        # Fallback to average of points if area is too small
        x_avg = sum(p.x for p in hull[:-1]) / (len(hull) - 1)
        y_avg = sum(p.y for p in hull[:-1]) / (len(hull) - 1)
        return SinglePoint(x=round_half_up(x_avg), y=round_half_up(y_avg))

    # Calculate final centroid coordinates
    centroid_x = centroid_x / (6.0 * area)
    centroid_y = centroid_y / (6.0 * area)

    return SinglePoint(x=round_half_up(centroid_x), y=round_half_up(centroid_y))


def shapely_to_polygon(
    polygon: ShapelyPolygon,
) -> Polygon:
    coords = list(polygon.exterior.coords)
    return Polygon(hull=[SinglePoint(round_half_up(x), round_half_up(y)) for x, y in coords])


def polygon_to_shapely(
    polygon: Polygon,
) -> ShapelyPolygon:
    return ShapelyPolygon([(p.x, p.y) for p in polygon.hull])


def ensure_polygon(polygon: ShapelyPolygon) -> ShapelyPolygon:
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda p: p.area)
    return polygon


def simplify_polygon_to_n_points(
    polygon: ShapelyPolygon | Polygon,
    target_num_points: int = 20,
    max_iter: int = 10,
    preserve_topology: bool = True,
) -> Polygon | None:
    """
    Iteratively adjusts the 'tolerance' parameter for shapely's simplify()
    so that the polygon ends up with approximately target_num_points exterior coordinates.
    """
    if isinstance(polygon, Polygon):
        polygon = polygon_to_shapely(polygon)
    coords = list(polygon.exterior.coords)
    if len(coords) <= target_num_points:
        return shapely_to_polygon(polygon)

    minx, miny, maxx, maxy = polygon.bounds
    diag_len = np.hypot((maxx - minx), (maxy - miny))

    low_tol = 0.0
    high_tol = diag_len
    best_poly = polygon
    best_diff = abs(len(coords) - target_num_points)

    for _ in range(max_iter):
        mid_tol = (low_tol + high_tol) * 0.5
        candidate = polygon.simplify(mid_tol, preserve_topology=preserve_topology)
        candidate = ensure_polygon(candidate)
        cand_coords = list(candidate.exterior.coords)
        diff = abs(len(cand_coords) - target_num_points)

        if diff < best_diff:
            best_diff = diff
            best_poly = candidate

        if len(cand_coords) > target_num_points:
            low_tol = mid_tol
        else:
            high_tol = mid_tol
        if best_diff == 0:
            break

    if not best_poly.is_valid:
        best_poly = best_poly.buffer(0)
    if not best_poly.is_valid:
        return None
    best_poly = ensure_polygon(best_poly)
    return shapely_to_polygon(best_poly)


def convert_binary_mask_to_polygons(
    binary_mask: np.ndarray,
    target_num_points: int = 20,
    contour_level: float = 0.5,
    simplification_tolerance: float = 2.0,
    polygon_filter_func: Optional[Callable[[ShapelyPolygon], bool]] = None,
) -> Optional[Polygon]:
    """
    Converts a binary mask to a polygon by:
      1) Finding contours at the specified level,
      2) Picking the largest contour by area,
      3) Simplifying the polygon using the given simplification tolerance,
      4) Iteratively adjusting to achieve roughly target_num_points.

    This version does not use any bounding box filtering.
    """
    contours = find_contours(binary_mask, level=contour_level)
    shapely_polygons = []

    for contour in contours:
        if len(contour) < 3:
            continue
        # Convert [row, col] (i.e. (y, x)) to (x, y)
        poly = ShapelyPolygon(contour[:, ::-1])
        if not poly.is_valid or poly.is_empty or poly.area == 0:
            continue

        if polygon_filter_func is None or polygon_filter_func(poly):
            poly_simpl = poly.simplify(simplification_tolerance, preserve_topology=True)
            shapely_polygons.append(poly_simpl)

    if not shapely_polygons:
        return None

    # Select the largest polygon (by area) among the candidates.
    largest_polygon = max(shapely_polygons, key=lambda p: p.area)

    # Further simplify the largest polygon to have roughly target_num_points.
    if target_num_points is not None:
        final_poly: Polygon = simplify_polygon_to_n_points(
            largest_polygon,
            target_num_points=target_num_points,
            max_iter=10,
            preserve_topology=True,
        )
    else:
        final_poly = shapely_to_polygon(largest_polygon)

    # Remove the repeated closing point if present.
    if len(final_poly.hull) > 1 and final_poly.hull[0] == final_poly.hull[-1]:
        final_poly.hull = final_poly.hull[:-1]

    if len(final_poly.hull) < 3:
        return None
    return final_poly


def box_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """
    Compute Intersection-over-Union between two BoundingBox objects.
    """
    # Extract coordinates from the first bounding box.
    xA1, yA1 = bbox1.top_left.x, bbox1.top_left.y
    xA2, yA2 = bbox1.bottom_right.x, bbox1.bottom_right.y

    # Extract coordinates from the second bounding box.
    xB1, yB1 = bbox2.top_left.x, bbox2.top_left.y
    xB2, yB2 = bbox2.bottom_right.x, bbox2.bottom_right.y

    # Compute the intersection width and height.
    inter_w = min(xA2, xB2) - max(xA1, xB1)
    inter_h = min(yA2, yB2) - max(yA1, yB1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0

    inter_area = inter_w * inter_h
    areaA = (xA2 - xA1) * (yA2 - yA1)
    areaB = (xB2 - xB1) * (yB2 - yB1)
    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def bbox_to_polygon(box: BoundingBox) -> Polygon:
    x1, y1 = box.top_left.x, box.top_left.y
    x2, y2 = box.bottom_right.x, box.bottom_right.y

    # define the four corners
    top_left = box.top_left
    top_right = SinglePoint(x2, y1)
    bottom_right = box.bottom_right
    bottom_left = SinglePoint(x1, y2)

    return Polygon(hull=[top_left, top_right, bottom_right, bottom_left], t=box.t)


def polygon_to_bbox(poly: Polygon) -> BoundingBox:
    """
    Returns the smallest bounding box that contains the polygon.
    """
    if not isinstance(poly, Polygon):
        raise ValueError("Input must be a Polygon")
    pts = poly.hull
    if not pts:
        raise ValueError("Empty polygon hull")
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    return BoundingBox(
        top_left=SinglePoint(min_x, min_y),
        bottom_right=SinglePoint(max_x, max_y),
    )

