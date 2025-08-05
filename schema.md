# MultiModal Data Schema Documentation

This document provides a comprehensive overview of our multimodal data schema system, designed to handle various content types including text, images, audio, video, and spatial pointing data.

## Core Concepts

The schema is built around these key concepts:

- **Content Types**: Different modalities supported (Text, Image, Audio, Video)
- **DataAtomic**: Base class for single-modal content
- **Document**: Container for multiple content pieces
- **Event**: Time-based content elements (used primarily in video)
- **Role**: Identifies the source of content (system, user, agent)
- **Point Types**: Spatial data representations (SinglePoint, BoundingBox, Polygon)

## Basic Models

### Role Enum

Defines the role of content within a conversation:

```json
{
  "role": "system" | "user" | "agent"
}
```

### ModalityType Enum

Defines the supported content types:

```json
{
  "type": "text" | "image" | "audio" | "video"
}
```

## Content Models

### Text

Basic text content:

```json
{
  "type": "text",
  "content": "Hello, world!",
  "metadata": {
    "language": "en",
    "confidence": 0.98
  },
  "role": "user"
}
```

### BranchText

Container for multiple text content options:

```json
{
  "type": "BranchText",
  "content": [
    {
      "type": "text",
      "content": "Option 1",
      "role": "agent"
    },
    {
      "type": "text",
      "content": "Option 2",
      "role": "agent"
    }
  ],
  "metadata": {
    "purpose": "dialog_options"
  }
}
```

### Image

Image content (base64 encoded):

```json
{
  "type": "image",
  "content": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
  "metadata": {
    "format": "png",
    "width": 1200,
    "height": 800
  },
  "role": "user"
}
```

### Audio

Audio content (base64 encoded):

```json
{
  "type": "audio",
  "content": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",
  "metadata": {
    "format": "wav",
    "duration_seconds": 30.5,
    "sample_rate": 44100
  },
  "role": "user"
}
```

### Video

Video content with tracks:

```json
{
  "type": "video",
  "content": "base64encodedvideodata...",
  "tracks": {
    "audio": [
      {
        "start_time_seconds": 0.0,
        "end_time_seconds": 10.5,
        "type": "audio",
        "content": {
          "type": "audio",
          "content": "base64encodedaudiodata..."
        }
      }
    ],
    "captions": [
      {
        "start_time_seconds": 0.0,
        "end_time_seconds": 5.2,
        "type": "text",
        "content": {
          "type": "text",
          "content": "Hello and welcome to our presentation."
        }
      }
    ]
  },
  "metadata": {
    "format": "mp4",
    "duration_seconds": 120.0,
    "resolution": "1920x1080"
  },
  "role": "agent"
}
```

### Document

Container for multiple content pieces:

```json
{
  "content": [
    {
      "type": "text",
      "content": "Document title",
      "metadata": {
        "format": "title"
      }
    },
    {
      "type": "image",
      "content": "base64encodedimagedata...",
      "metadata": {
        "description": "Cover image"
      }
    },
    {
      "type": "text",
      "content": "Document body text goes here...",
      "metadata": {
        "format": "body"
      }
    }
  ],
  "metadata": {
    "document_id": "doc_12345",
    "created_at": "2025-01-15T12:00:00Z"
  }
}
```

## Python Usage Examples

### Creating Text Content

```python
from schema import Text, Role, ModalityType

# Create a simple text object
text = Text(
    content="Hello, world!",
    role=Role.USER,
    metadata={"language": "en"}
)

# Access properties
print(text.content)  # "Hello, world!"
print(text.type)     # "text"
print(text.role)     # Role.USER
```

### Creating BranchText

```python
from schema import BranchText, Text, Role

# Create multiple text options
options = BranchText(
    content=[
        Text(content="Yes, I'd like to proceed", role=Role.AGENT),
        Text(content="No, let's try something else", role=Role.AGENT),
        Text(content="Tell me more information", role=Role.AGENT)
    ],
    metadata={"context": "user_confirmation"}
)
```

### Creating an Image

```python
from schema import Image, Role
import base64

# Load image file and encode to base64
with open("image.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# Create image object
image = Image(
    content=encoded_image,
    role=Role.USER,
    metadata={"width": 800, "height": 600, "format": "png"}
)
```

### Creating a Video with Tracks

```python
from schema import Video, Audio, Text, Event, ModalityType, Role

# Create a video with multiple tracks
video = Video(
    content="base64encodedvideodata...",
    role=Role.AGENT,
    tracks={
        "audio": [
            Event(
                start_time_seconds=0.0,
                end_time_seconds=30.0,
                type=ModalityType.AUDIO,
                content=Audio(
                    content="base64encodedaudiodata...",
                    metadata={"bitrate": "320kbps"}
                )
            )
        ],
        "captions": [
            Event(
                start_time_seconds=0.0,
                end_time_seconds=5.0,
                type=ModalityType.TEXT,
                content=Text(content="Welcome to our video")
            ),
            Event(
                start_time_seconds=5.1,
                end_time_seconds=10.0,
                type=ModalityType.TEXT,
                content=Text(content="Let's explore this topic...")
            )
        ]
    },
    metadata={
        "duration_seconds": 30.0,
        "resolution": "1920x1080",
        "format": "mp4"
    }
)
```

### Creating a Complete Document

```python
from schema import Document, Text, Image, Role, ModalityType

# Create a document with multiple content types
document = Document(
    content=[
        Text(
            content="Quarterly Report Q1 2025",
            role=Role.SYSTEM,
            metadata={"format": "title"}
        ),
        Text(
            content="Executive Summary: Our company has shown strong growth in Q1...",
            role=Role.SYSTEM,
            metadata={"format": "summary"}
        ),
        Image(
            content="base64encodedimagedata...",
            role=Role.SYSTEM,
            metadata={"description": "Q1 Sales Graph"}
        ),
        Text(
            content="Detailed Analysis: The sales team exceeded targets by 15%...",
            role=Role.SYSTEM,
            metadata={"format": "body"}
        )
    ],
    metadata={
        "author": "Finance Team",
        "created_at": "2025-04-05T09:00:00Z",
        "document_id": "report_q1_2025"
    }
)

# Convert to JSON for frontend consumption
document_json = document.model_dump_json(indent=2)
```

## Validation Rules

The schema enforces several validation rules:

1. **Base64 validation** for binary content (images, audio, video)
2. **Type consistency** between Event types and their content
3. **Time validation** in Events (end time must be >= start time)
4. **Content type matching** (e.g., Text content must be strings)

### Example: Base64 Validation

```python
from schema import Image

# This will raise a ValueError due to invalid base64
try:
    image = Image(content="not-valid-base64!")
except ValueError as e:
    print(f"Validation error: {e}")
```

### Example: Event Time Validation

```python
from schema import Event, Text, ModalityType

# This will raise a ValueError (end time before start time)
try:
    event = Event(
        start_time_seconds=10.0,
        end_time_seconds=5.0,  # Error: less than start_time
        type=ModalityType.TEXT,
        content=Text(content="This caption won't validate")
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

## Best Practices

1. **Always validate input** before creating model instances
2. **Use proper encoding** for binary content (images, audio, video)
3. **Include meaningful metadata** to assist frontend rendering
4. **Check role consistency** across related content pieces
5. **Be careful with large binary content** - consider client-side loading strategies for better performance

## Point Data Models

The schema includes powerful spatial data representations used for locating or annotating regions in images. Point data is semantically connected to image content and represents spatial references within those images.

### Semantic Context

Points in this schema are designed to have a semantic relationship with images:

- **Reference Context**: Points typically refer to the most recently mentioned image in a conversation
- **Spatial Mapping**: Coordinates are relative to the image dimensions (origin at top-left)
- **Content Association**: Points identify regions of interest, objects, or features within the referenced image
- **Conversation Flow**: In a conversation, point data (from agent or user) implicitly references the previously shared image without needing explicit linking

This semantic connection allows for natural interaction patterns such as:

1. User shares an image
2. User requests information about something in the image
3. Agent responds with point data to indicate specific locations or regions
4. The point data is implicitly understood to reference the previously shared image

### Point Types Hierarchy

```
Point (base class)
├── SinglePoint
├── BoundingBox
└── Polygon
```

### SinglePoint

Represents a single coordinate in 2D space:

```json
{
  "x": 150,
  "y": 225
}
```

### BoundingBox

Defines a rectangular region with top-left and bottom-right points:

```json
{
  "top_left": {
    "x": 100,
    "y": 100
  },
  "bottom_right": {
    "x": 300,
    "y": 250
  }
}
```

### Polygon

Represents an arbitrary shape through a series of connected points:

```json
{
  "hull": [
    { "x": 100, "y": 100 },
    { "x": 200, "y": 50 },
    { "x": 300, "y": 100 },
    { "x": 250, "y": 200 },
    { "x": 150, "y": 200 }
  ]
}
```

## Pointing Functionality

The schema provides utilities for parsing, normalizing, and working with spatial point data.

### Point String Representation

Points can be represented in a string format with XML-style tags:

```
<point>(150,225)</point>               # SinglePoint
<point_box>(100,100) (300,250)</point_box>  # BoundingBox
<polygon>(100,100) (200,50) (300,100) (250,200) (150,200)</polygon>  # Polygon
```

### PointParser

Utility for converting between point objects and string representations:

```python
from genesis.data.pointing.parsers import HTMLPointParser, 
from genesis.data.pointing.pointing import SinglePoint, BoundingBox, Polygon

# Create a point string from a point object
point = SinglePoint(x=150, y=225)
point_str = HTMLPointParser.serialize(point)  # "<point>(150,225)</point>"

# Parse a point string into a point object
box_str = "<point_box>(100,100) (300,250)</point_box>"
box_obj = HTMLPointParser.parse(box_str)  # BoundingBox instance
```

### Normalization and Denormalization

Convert between pixel coordinates and normalized coordinates:

```python
from genesis.data.pointing import NormalizationConfig, SinglePoint

# Define normalization configuration
config = NormalizationConfig(
    range_x=100,  # Normalized range for x
    range_y=100,  # Normalized range for y
    image_width=1920,  # Original image width
    image_height=1080   # Original image height
)

# Normalize a point from pixel space to normalized space
pixel_point = SinglePoint(x=960, y=540)  # Center of 1920x1080 image
norm_point = pixel_point.normalize(config)  # SinglePoint(x=50, y=50)

# Denormalize a point from normalized space to pixel space
norm_point = SinglePoint(x=25, y=75)
pixel_point = norm_point.denormalize(config)  # SinglePoint(x=480, y=810)
```

## Python Examples for Point Data

### Creating and Using SinglePoint

```python
from genesis.data.pointing import SinglePoint

# Create a point
point = SinglePoint(x=150, y=225)

# Convert to string representation
point_str = point.to_point_string()  # "(150,225)"

# Parse from string
parsed_point = SinglePoint.parse_point_string("(150,225)")
```

### Working with BoundingBox

```python
from genesis.data.pointing import BoundingBox, SinglePoint

# Create a bounding box
box = BoundingBox(
    top_left=SinglePoint(x=100, y=100),
    bottom_right=SinglePoint(x=300, y=250)
)

# Get string representation
box_str = box.to_point_string()  # "(100,100) (300,250)"

# Calculate IoU (Intersection over Union) between two boxes
from genesis.data.pointing import box_iou

box1 = BoundingBox(
    top_left=SinglePoint(x=100, y=100),
    bottom_right=SinglePoint(x=300, y=300)
)
box2 = BoundingBox(
    top_left=SinglePoint(x=200, y=200),
    bottom_right=SinglePoint(x=400, y=400)
)
iou = box_iou(box1, box2)  # Returns a value between 0.0 and 1.0
```

### Working with Polygon

```python
from genesis.data.pointing import Polygon, SinglePoint, polygon_to_centroid

# Create a polygon (rectangle)
polygon = Polygon(hull=[
    SinglePoint(x=100, y=100),
    SinglePoint(x=300, y=100),
    SinglePoint(x=300, y=200),
    SinglePoint(x=100, y=200)
])

# Get string representation
poly_str = polygon.to_point_string()  # "(100,100) (300,100) (300,200) (100,200)"

# Calculate centroid
centroid = polygon_to_centroid(polygon)  # SinglePoint(x=200, y=150)
```

### Converting Between Polygon and Shapely Polygon

```python
from genesis.data.pointing import Polygon, SinglePoint, polygon_to_shapely, shapely_to_polygon
from shapely.geometry import Polygon as ShapelyPolygon

# Create a polygon
point_polygon = Polygon(hull=[
    SinglePoint(x=0, y=0),
    SinglePoint(x=100, y=0),
    SinglePoint(x=100, y=100),
    SinglePoint(x=0, y=100)
])

# Convert to Shapely polygon
shapely_poly = polygon_to_shapely(point_polygon)

# Perform operations with Shapely
simplified_shapely = shapely_poly.simplify(tolerance=1.0)

# Convert back to our Polygon format
simplified_polygon = shapely_to_polygon(simplified_shapely)
```

### Converting Binary Masks to Polygons

```python
import numpy as np
from genesis.data.pointing import convert_binary_mask_to_polygons

# Create a binary mask (e.g., from semantic segmentation)
mask = np.zeros((100, 100), dtype=np.uint8)
mask[25:75, 25:75] = 1  # Create a square in the middle

# Convert to polygon
polygon = convert_binary_mask_to_polygons(
    binary_mask=mask,
    target_num_points=20,  # Target number of points for simplification
    contour_level=0.5
)
```

### Simplifying Polygons

```python
from genesis.data.pointing import Polygon, SinglePoint, simplify_polygon_to_n_points

# Create a complex polygon with many points
complex_polygon = Polygon(hull=[...])  # Many points

# Simplify to a target number of points
simplified_polygon = simplify_polygon_to_n_points(
    polygon=complex_polygon,
    target_num_points=20,
    max_iter=10
)
```

## Integration with Documents

You can include point data in Document content, with points semantically referring to previously shared images:

```python
from genesis.data.schema import Document, Text, Image, Role
from genesis.data.pointing.parsers import HTMLPointParser
from genesis.data.pointing.pointing import SinglePoint
import base64

# Create a document with an image followed by point reference
document = Document(
    content=[
        # First, include an image in the conversation
        Image(
            content=base64.b64encode(open("cat_image.jpg", "rb").read()).decode("utf-8"),
            role=Role.USER
        ),

        # User asks about something in the image
        Text(
            content="Can you identify where the cat's eye is in this picture?",
            role=Role.USER
        ),

        # Agent response with point data that semantically refers to the image
        Text(
            content=f"The cat's eye is located at {HTMLPointParser.serialize(SinglePoint(x=150, y=225))}",
            role=Role.AGENT
        )
    ]
)
```

### Conversation Flow with Implicit Image References

The schema supports natural conversation flows where points reference the context image:

```python
# Example conversation with image reference
conversation = Document(
    content=[
        # Image shared
        Image(content="base64_encoded_image_data", role=Role.USER),

        # User request
        Text(content="What's in this box?", role=Role.USER),

        # Agent response with bounding box that implicitly refers to the image
        Text(
            content=f"I can see a cat in this box: {HTMLPointParser.serialize(BoundingBox(
                top_left=SinglePoint(x=100, y=100),
                bottom_right=SinglePoint(x=300, y=250)
            ))}",
            role=Role.AGENT
        ),

        # User follow-up
        Text(content="Can you outline just the cat's face?", role=Role.USER),

        # Agent response with polygon that implicitly refers to the same image
        Text(
            content=f"Here's the outline of the cat's face: {HTMLPointParser.serialize(Polygon(hull=[
                SinglePoint(x=150, y=120),
                SinglePoint(x=250, y=120),
                SinglePoint(x=275, y=175),
                SinglePoint(x=200, y=220),
                SinglePoint(x=125, y=175)
            ]))}",
            role=Role.AGENT
        )
    ]
)
```

### Extracting Points from Conversations

```python
from genesis.data.schema import Role, ModalityType
from genesis.evals.pointing.utils import extract_content

# Extract all points created by the agent from a conversation
points = extract_content(
    conversation=conversation,
    role=Role.AGENT,
    modality_type=ModalityType.TEXT,
    processing_function=lambda x: x  # Optional processing function
)

# Process extracted points
for point_text in points:
    try:
        # Parse the point string to get the actual point object
        point_obj = HTMLPointParser.parse(point_text)
        print(f"Found point of type: {point_obj.__class__.__name__}")
    except Exception as e:
        print(f"Error parsing point: {e}")
```

## Troubleshooting

Common issues and solutions:

- **"Not valid Base64" errors**: Ensure proper encoding and padding
- **Type mismatches**: Verify that content type matches the declared type
- **Invalid time ranges**: Check that end times are greater than or equal to start times
- **Point parsing errors**: Ensure correct formatting of point strings (parentheses, commas, spaces)
- **Performance issues**: Consider chunking large binary content or using streaming approaches
- **Polygon simplification**: Adjust target points and tolerance for better results

For more assistance, consult the API documentation or reach out to the backend team.
