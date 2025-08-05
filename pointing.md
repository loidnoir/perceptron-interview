# Structured Points for Perceptron

The goal of structured points in our data classes is to standardize our representation of all “points.” A point can be:

- A single x,y coordinate (`SinglePoint`)
- A bounding box consisting of top-left and bottom-right points (`BoundingBox`)
- A polygon defined by a list of points forming the hull (`Polygon`)

We should be able to represent any structured point through these dataclasses and use our standardized tools for simplification and serialization.

## Steps to Create a Point
1. Construct a point by parsing dataset format into our `Point` class (see **Structured Point Class**).
2. If working with segmentation masks or polygons, apply **Polygon Simplification**.
3. Normalize points into a 1000x1000 grid on images (see **Normalization**).
4. Store the `normalization_config` in the metadata.

---

## Structured Point Class
**File:** `data/pointing.py`  

A simplified definition of the dataclasses:

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Point:
    """Base class for point types"""
    pass

@dataclass
class SinglePoint(Point):
    """A single point with x,y coordinates

    Example:
        point = SinglePoint(1, 2)
    """

    x: int
    y: int
    # temporal dimension to refer to a specific
    # image or timestamp in a video
    t: float | None = None

@dataclass
class BoundingBox(Point):
    """A bounding box defined by top-left and bottom-right points

    Example:
        box = BoundingBox(SinglePoint(1,1), SinglePoint(5,5))
    """

    top_left: SinglePoint
    bottom_right: SinglePoint
    # temporal dimension to refer to a specific
    # image or timestamp in a video
    t: float

@dataclass
class Polygon(Point):
    """A mask defined by a list of points forming a hull/polygon

    Example:
        mask = Polygon([SinglePoint(1,1), SinglePoint(2,2), SinglePoint(3,1)])
    """

    hull: List[SinglePoint]
    # temporal dimension to refer to a specific
    # image or timestamp in a video
    t: float
```

# Serializing Points

Once we have points constructed, we use a utility `PointParser` to convert `Point` objects into strings and reconstruct them from strings. This ensures a standardized format for data storage and transfer.

An example for polygons

```python
from genesis.data.pointing.parsers import HTMLPointParser

parsed_mask: Polygon = HTMLPointParser.parse("<polygon> (4,4) (5,5) (6,4) </polygon>")
mask_str = HTMLPointParser.serialize(parsed_mask)
self.assertEqual(mask_str, "<polygon> (4,4) (5,5) (6,4) </polygon>")
```

# Normalization
We standardize all of our points to a grid of 1000 by 1000 where each coordinate is represented as integers, following [Qwen2VL (Sec 2.2.1)](https://arxiv.org/pdf/2409.12191) and [Gemini-2 Bounding Box](https://cloud.google.com/vertex-ai/generative-ai/docs/bounding-box-detection). Certain datasets follow different normalization schemes (e.g. 100 x 100, pixel coordinates, etc). 

We account for this by defining a NormalizationConfig and storing the normalized points. All the detail in the normalization config makes it easy to recover what the points should be.

An example for how we normalize:

```python
# assume points are pixel level coordinates
point = Polygon(hull=[SinglePoint(125, 40000), SinglePoint(....)])

# define a normalization config for 1000 by 1000
normalization_config = NormalizationConfig(
  range_x=1000,
  range_y=1000,
  image_width=width,
  image_height=height,
)

point = point.normalize(config=normalization_config)
```

Critically we store the normalization config in the metadata for each message to be able to reconstruct any of the points from the image.

# [Polygon Simplification](./polygon_simplification.md)
When working with polygons datasets often represent polygons as hundreds of points or as binary masks, we have a set of utilities that can convert either into a simplified polygon. Empirically we found a polygon with 20 points can represent random samples of binary masks and larger polygon (100+ points) with > 95 IOU.  So we chose to target a **maximum of ~20 points per polygon.**

## Simplifying a Polygon to N Point

We support the following function to convert an polygon object into one that is close to N points by searching for the best approximation 

```python
def simplify_polygon_to_n_points(
    polygon: ShapelyPolygon | Polygon,
    target_num_points: int = 20,
    max_iter: int = 10,
    preserve_topology: bool = True,
) -> Polygon:
```

## Binary Mask to Polygon + Simplification
Given a binary mask representing an object, the below function converts the binary masks into polygons, note that if we fail to  convert a polygon this will return None and only a single polygon will be returned (representing the largest object). 

If you have multiple objects segmented (e.g. all people marked as True) you should pass in only one object at a time.

```python
def convert_binary_mask_to_polygons(
    binary_mask: np.ndarray,
    target_num_points: int = 20,
    contour_level: float = 0.5,
    simplification_tolerance: float = 2.0,
    polygon_filter_func: Optional[Callable[[ShapelyPolygon], bool]] = None,
) -> Optional[Polygon]:
```