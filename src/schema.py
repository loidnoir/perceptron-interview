import base64
import io
import re
import warnings
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Counter, Dict, Literal, Union

import numpy as np
from pydantic import (AfterValidator, BaseModel, Field, ValidationInfo,
                      model_validator)
from pydantic_extra_types.semantic_version import SemanticVersion

from src.parser import HTMLPointParser, PointParsingError
from src.pointing import NOT_A_POINT, is_valid_normalization_config


def validate_base64_value(content: str | bytes) -> str:
    """
    Validate that the given content is a valid Base64-encoded string.
    Converts content to str if it's bytes, then checks:
      - content length is a multiple of 4
      - all characters in the allowed Base64 charset
      - decodes successfully via base64
    Raises ValueError if invalid; returns content (as a str) if valid.
    """
    # If content is in bytes, decode it to str
    if isinstance(content, bytes):
        content = content.decode("utf-8")

    # Ensure the final content is a string
    if not isinstance(content, str):
        raise ValueError("Content must be a string.")

    # Basic length check
    if len(content) % 4 != 0:
        raise ValueError("Not valid Base64 (length must be multiple of 4).")

    # Allowed Base64 character set
    allowed_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="

    # Check characters
    if not set(content).issubset(allowed_chars):
        raise ValueError("Not valid Base64 (contains invalid characters).")

    # Attempt to decode
    try:
        base64.b64decode(content)
    except Exception as err:
        raise ValueError("Not valid Base64 (decoding failed).") from err

    return content


def validate_base64_value_nullable(
    content: str | bytes | None,
) -> str | None:
    """
    Like validate_base64_value, but gracefully handles None
    (i.e., skip validation and just return None).
    """
    if content is None:
        return None
    return validate_base64_value(content)


def validate_mask_pixel_values(content: str | None) -> str | None:
    """
    Validates that a base64-encoded image mask has a reasonable number of unique pixel values.

    For segmentation masks, we expect a limited number of distinct values representing
    different segments/objects. This validator ensures the mask has <= 256 unique pixel values.

    Args:
        content: Base64-encoded image data or None

    Returns:
        The original content if valid, None if content is None

    Raises:
        ValueError: If the mask has too many unique pixel values or other validation errors
    """
    if content is None:
        return None

    # First validate it's proper base64
    content = validate_base64_value(content)

    # If content is empty string, log warning and allow it for backward compatibility
    # but this is not expected for ImageMask
    if not content:
        warnings.warn("ImageMask has empty content - this is unexpected as masks should contain image data")
        return content

    try:
        from PIL import Image as PILImage

        image_bytes = base64.b64decode(content)
        image = PILImage.open(io.BytesIO(image_bytes))
        image_array = np.array(image)

        # Count unique pixel values
        if len(image_array.shape) == 3:
            # Reshape to (height*width, channels) and get unique rows
            reshaped = image_array.reshape(-1, image_array.shape[-1])
            unique_pixels = np.unique(reshaped, axis=0)
            unique_count = len(unique_pixels)
        else:  # Grayscale image
            unique_count = len(np.unique(image_array))

        # Validate pixel count - limit of 256 for now
        if unique_count > 256:
            raise ValueError(
                f"ImageMask has too many unique pixel values ({unique_count}). Expected <= 256 for segmentation masks."
            )

        return content

    except ImportError as e:
        # If PIL/numpy not available, skip pixel validation but warn
        warnings.warn(f"Skipping pixel validation due to missing dependency: {e}")
        return content
    except ValueError as e:
        # Re-raise ValueError for pixel count validation (this should fail validation)
        if "too many unique pixel values" in str(e):
            raise e
        else:
            # Other ValueError (like image parsing issues) should warn but allow
            warnings.warn(f"Could not validate ImageMask pixel values (content may not be a valid image): {e}")
            return content
    except Exception as e:
        # If we can't parse as an image, log warning but allow it for backward compatibility
        # This covers cases where base64 content is valid but not an image (e.g., text data)
        warnings.warn(f"Could not validate ImageMask pixel values (content may not be a valid image): {e}")
        return content


class Role(Enum):
    """
    Enum representing the role of the content within a conversation.
    """

    SYSTEM = "system"
    USER = "user"
    AGENT = "agent"


class ModalityType(str, Enum):
    """
    Enum representing different content modalities.
    """

    TEXT = "text"
    IMAGE = "image"
    IMAGE_MASK = "image_mask"
    AUDIO = "audio"
    VIDEO = "video"
    REASONING = "reasoning"


class DataAtomic(BaseModel):
    metadata: dict[str, Any] | None = None
    role: Role | None = None

    def __str__(self) -> str:
        """
        A more readable string representation that:
          - Decodes bytes to string if necessary
          - Truncates long content to 50 chars
        """
        if isinstance(self.content, bytes):
            content_str = self.content.decode("utf-8", errors="replace")
        else:
            content_str = str(self.content)

        # Truncate if too long
        max_len = 50
        if len(content_str) > max_len:
            content_str = content_str[:max_len] + "..."

        return (
            f"{self.__class__.__name__}(type={self.type}, "
            f"content={content_str!r}, "
            f"metadata={self.metadata}, "
            f"role={self.role})"
        )

    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))


class Text(DataAtomic):
    type: Annotated[Literal[ModalityType.TEXT], Field(frozen=True)] = ModalityType.TEXT
    content: str | None = None  # For text modality, content must be a string

    def __str__(self) -> str:
        """Override to show full text content without truncation."""
        return (
            f"{self.__class__.__name__}(type={self.type}, "
            f"content={self.content!r}, "
            f"metadata={self.metadata}, "
            f"role={self.role})"
        )


class Reasoning(DataAtomic):
    type: Annotated[Literal[ModalityType.REASONING], Field(frozen=True)] = ModalityType.REASONING
    content: str | None = None  # For reasoning modality, content must be a string

    # We need an invariant that reasoning is always agent
    @model_validator(mode="after")
    def validate_role(self) -> "Reasoning":
        if self.role != Role.AGENT:
            raise ValueError("Reasoning must always have role 'agent'")
        return self


class BranchText(BaseModel):
    type: Annotated[Literal["BranchText"], Field(frozen=True)] = "BranchText"
    content: list[Text] = Field(min_length=1)
    metadata: dict[str, Any] | None = None

    def __str__(self) -> str:
        # Example of customizing if you want a nice display for BranchText as well
        return f"BranchText(type={self.type}, content=[{', '.join(str(t) for t in self.content)}])"


class Image(DataAtomic):
    type: Annotated[Literal[ModalityType.IMAGE], Field(frozen=True)] = ModalityType.IMAGE
    content: Annotated[str | None, AfterValidator(validate_base64_value_nullable)] = None
    metadata: dict[str, Any] | None = None


class ImageMask(DataAtomic):
    """
    Represents a masked image where certain regions are highlighted or segmented.

    These images are meant to be converted into point coordinates for downstream processing,
    rather than containing pointing markup themselves.

    ImageMask always accompanies an Image, which provides the original image data.
    The ordering constraint is: content=[...,Image,ImageMask,...] where the prior Image
    is referred to by the latter ImageMask.
    """

    type: Annotated[Literal[ModalityType.IMAGE_MASK], Field(frozen=True)] = ModalityType.IMAGE_MASK
    content: Annotated[str | None, AfterValidator(validate_mask_pixel_values)] = None
    metadata: dict[str, Any] | None = None


class Audio(DataAtomic):
    type: Annotated[Literal[ModalityType.AUDIO], Field(frozen=True)] = ModalityType.AUDIO
    content: Annotated[str | None, AfterValidator(validate_base64_value_nullable)] = None
    metadata: dict[str, Any] | None = None


Tracks = dict[str, list["Event"]]


class Video(DataAtomic):
    type: Annotated[Literal[ModalityType.VIDEO], Field(frozen=True)] = ModalityType.VIDEO
    content: Annotated[str | None, AfterValidator(validate_base64_value_nullable)] = None
    tracks: Tracks
    metadata: dict[str, Any] | None = None

    def __str__(self) -> str:
        # grab the data atomic string representation (e.g. "Video(type=…, content=…, metadata=…, role=…)")
        base = super().__str__().rstrip(")")
        # append the tracks, then re-close the parenthesis
        return f"{base}, tracks={self.tracks})"


class Reference(BaseModel):
    type: Literal["reference"] = Field(default="reference", frozen=True)
    index: int

    @model_validator(mode="after")
    def check_index(self):
        if self.index < 0:
            raise ValueError(f"Reference index must be ≥ 0, got {self.index}")
        return self


class Branch(BaseModel):
    type: Literal["branch"] = Field(default="branch", frozen=True)
    content: list[list["Content"]] = Field(min_length=1)
    metadata: dict[str, Any] | None = None

    # ---------- pre-validators -------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def accept_flat_or_nested(cls, values: dict):
        """
        Allow callers to supply either
            • a *flat* list  [Text(), Reference(), …]         (legacy form)
            • or the new     [[Text(), …], [Text(), …], …].

        If we detect the flat form we simply wrap it once so downstream
        validation continues to work.
        """
        cont = values.get("content")
        if cont and cont and not isinstance(cont[0], list):
            # flat → nested
            values["content"] = [cont]

        if any(
            isinstance(item, dict) and item.get("type") == "BranchText"
            for item in values["content"]
            for item in (item if isinstance(item, list) else [item])
        ):
            raise ValueError("BranchText cannot be nested inside another BranchText")

        return values

    # ---------- post-validators ------------------------------------------------
    @model_validator(mode="after")
    def no_empty_paths(self):
        if any(len(path) == 0 for path in self.content):
            raise ValueError("Each branch alternative must contain at least one item")
        return self


# Define the discriminated union for content
Content = Annotated[
    Union[
        BranchText,
        Text,
        Image,
        ImageMask,
        Audio,
        Video,
        Reasoning,
        Reference,
        Branch,
    ],
    Field(discriminator="type"),
]


class Event(DataAtomic):
    """
    Represents a timed event within a larger piece of content (e.g., in a Video track).
    Events require content to be Text, else validation will fail.
    """

    start_time_seconds: float
    end_time_seconds: float
    content: Content

    @model_validator(mode="before")
    @classmethod
    def ignore_type_field(cls, values: dict) -> dict:
        """Ignore the 'type' field if present in stored data for backward compatibility."""
        if isinstance(values, dict) and "type" in values:
            values.pop("type")
        return values

    @model_validator(mode="after")
    def validate_times(self) -> "Event":
        """
        Validates that time values are numeric and end_time >= start_time
        """
        # Validate time values
        if not isinstance(self.start_time_seconds, int | float) or not isinstance(self.end_time_seconds, int | float):
            raise ValueError("Time values must be numbers.")

        if self.end_time_seconds < self.start_time_seconds:
            raise ValueError("End time must be greater than or equal to start time.")

        return self

    @model_validator(mode="after")
    def validate_agent_event_is_text(self) -> "Event":
        """
        Validates that agent events are Text.
        """
        if self.role == Role.AGENT and not isinstance(self.content, Text):
            raise ValueError("Agent events must have Text content")
        return self

    def __str__(self) -> str:
        return f"Event(start={self.start_time_seconds}, end={self.end_time_seconds}, content={self.content})"


def text_to_point_counts(text_content_item: Text) -> Counter:
    """
    Extracts and counts valid XML point tags (<point>, <point_box>, or <polygon>)
    from anywhere in the text.

    Returns:
        A Counter where keys are the point type names (e.g. "point", "point_box", "polygon")
        and values are the counts of how many times that type appears.

    If no valid point tag is found (or if the text is empty), returns Counter({NOT_A_POINT: 1}).
    """
    if not hasattr(text_content_item, "content"):
        return Counter({NOT_A_POINT: 1})
    text = text_content_item.content
    if not text:
        return Counter({NOT_A_POINT: 1})

    pattern = r"<(point(?:_box)?|polygon)(?:\s[^>]*)?>(.*?)</\1>"
    matches = list(re.finditer(pattern, text, flags=re.DOTALL))
    if not matches:
        return Counter({NOT_A_POINT: 1})
    counts = Counter()
    for match in matches:
        xml_snippet = match.group(0)
        try:
            # Attempt to parse the XML snippet to get a point object.
            point_obj = HTMLPointParser.parse(xml_snippet)
            point_type = point_obj.__class__.point_name
            counts[point_type] += 1
        except PointParsingError:
            # Point formatting but failed point class creation
            pass
    return counts


class Document(BaseModel):
    """
    A container for multiple pieces of content (DataAtomic).
    """

    type: Annotated[Literal["Document"], Field(frozen=True)] = "Document"
    schema_version: SemanticVersion = Field(default=SemanticVersion.parse("2.0.0"), frozen=True)
    references: list[Union[Text, Image, ImageMask, Audio, Video, Reasoning]] = Field(default_factory=list)
    content: list[Content]
    metadata: dict[str, Any] | None = None

    def __hash__(self):
        hash_value = 0
        for c in self.content:
            hash_value += hash(c)
        return hash_value

    def __str__(self) -> str:
        return "Document(content=[\n  " + ",\n  ".join(str(c) for c in self.content) + f"\n], metadata={self.metadata})"

    @model_validator(mode="before")
    @classmethod
    def migrate_branchtext(cls, values: Any) -> Any:
        if isinstance(values, dict):
            ver = str(values.get("schema_version", "0.0.0"))
            if ver.startswith(("0.", "1.")) and "content" in values:
                new_content = []
                for itm in values.get("content", []):
                    if isinstance(itm, dict) and itm.get("type") == "BranchText":
                        if isinstance(itm["content"], list):
                            # BranchText can contain a list of Texts, which are supposed to be individual alternatives.
                            # We need to wrap each of these elements in a list.
                            new_content.append(
                                {
                                    "type": "branch",
                                    "content": [[c] for c in itm["content"]],
                                }
                            )
                        else:
                            new_content.append(
                                {
                                    "type": "branch",
                                    "content": [itm["content"]],
                                }
                            )
                    else:
                        new_content.append(itm)
                values["content"] = new_content
        return values

    @model_validator(mode="after")
    def validate_refs_and_branches(self):
        n = len(self.references)

        def walk(itm):
            if isinstance(itm, Reference):
                if not (0 <= itm.index < n):
                    raise ValueError(f"Reference index {itm.index} out of range 0–{n - 1}")
            elif isinstance(itm, Branch):
                for path in itm.content:
                    for child in path:
                        walk(child)

        for top in self.content:
            walk(top)
        return self

    @model_validator(mode="after")
    def validate_image_mask_ordering(self) -> "Document":
        """
        Validate that ImageMask elements come after their corresponding Image elements.
        The ordering constraint is: content=[...,Image,ImageMask,...] where the prior Image
        is referred to by the latter ImageMask.
        """
        for i, item in enumerate(self.content):
            if isinstance(item, ImageMask):
                # Find the most recent Image before this ImageMask
                found_image = False
                for j in range(i - 1, -1, -1):
                    if isinstance(self.content[j], Image):
                        found_image = True
                        break

                if not found_image:
                    raise ValueError(
                        f"ImageMask at content index {i} must be preceded by an Image. "
                        f"The ordering constraint is: content=[...,Image,ImageMask,...]"
                    )

        return self

    @model_validator(mode="after")
    def check_normalization_config_if_points_exist(self, info: ValidationInfo) -> "Document":
        """
        Validate NormalizationConfig presence if points are detected in Text content.
        """
        if info.context and info.context.get("skip_normalization_check"):
            return self

        has_points = False
        for item in self.content:
            if isinstance(item, Text):
                counts = text_to_point_counts(item)
                # Check if the counter is not empty and doesn't just contain NOT_A_POINT
                if counts and counts != Counter({NOT_A_POINT: 1}):
                    has_points = True
                    break  # Found points, no need to check further

        if not has_points:
            return self  # No points found, no NormalizationConfig needed so we early exit

        # --- Points were found, proceed with config checks ---

        # 1. Check all Images for a NormalizationConfig
        # Note: ImageMasks are not included here because they represent images that are
        # meant to be converted INTO points, not images that contain pointing markup
        found_an_image = False
        for idx, item in enumerate(self.content):
            if isinstance(item, Image):
                found_an_image = True
                image_config_data = None
                if item.metadata:
                    # Recursive function to search for normalization_config in nested dictionaries
                    def find_normalization_config(metadata_dict):
                        if not isinstance(metadata_dict, dict):
                            return None

                        # Check if normalization_config exists at this level
                        if "normalization_config" in metadata_dict:
                            return metadata_dict["normalization_config"]

                        # Recursively search in nested dictionaries
                        for value in metadata_dict.values():
                            if isinstance(value, dict):
                                result = find_normalization_config(value)
                                if result is not None:
                                    return result

                        return None

                    image_config_data = find_normalization_config(item.metadata)
                else:
                    image_config_data = None
                if not is_valid_normalization_config(image_config_data):
                    # Found an image missing the required config
                    raise ValueError(
                        f"Document contains points, but the Image at content index {idx} "
                        f"is missing a valid 'normalization_config' in its metadata."
                    )

        # 2. If we reach this section, either all images had a NormalizationConfig, or there were no images
        if not found_an_image:
            raise ValueError(
                "Document contains points, but no Image items were present to provide a "
                "'normalization_config' in their metadata. Note: ImageMasks are not valid "
                "for pointing as they represent processed images meant to be converted into points."
            )

        return self  # All images had a NormalizationConfig

    def to_json_dict(self) -> Dict[str, Any]:
        """Return a fully JSON-serialisable representation of the ``Document``.

        Conversion rules
        ----------------
        * Objects exposing ``model_dump()``/``dict()`` are recursively serialised.
        * ``bytes``/``bytearray`` are base-64-encoded (loss-less for raw media).
        * ``Enum``/``SemanticVersion``/``Path`` → their string value.
        * Built-in scalars (``str``, ``int``, ``float``, ``bool``, ``None``) pass
        through unchanged.
        * Containers (``Mapping``, ``Sequence``) are handled recursively.
        * Unknown types fall back to ``str(obj)`` so the method never raises.
        """

        def _encode(obj: Any) -> Any:  # noqa: ANN401
            # 1️ Pydantic (& similar) models
            if hasattr(obj, "model_dump"):
                return _encode(obj.model_dump())

            # 2️ Simple JSON-native scalars
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj

            # 3️ Binary blobs
            if isinstance(obj, (bytes, bytearray)):
                return base64.b64encode(obj).decode("ascii")

            # 4️ Path-like & enum-like objects
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, Path):
                return str(obj)

            # 5️ Containers
            if isinstance(obj, (list, tuple)):
                return [_encode(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _encode(v) for k, v in obj.items()}

            # 6️ Fallback – guarantees we never crash
            return str(obj)

        return {
            "schema_version": str(self.schema_version),
            "references": _encode(self.references),
            "content": _encode(self.content),
            "metadata": _encode(self.metadata),
        }
