import re
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable

from .pointing import (BoundingBox, Point, Polygon, SinglePoint,
                       bbox_to_polygon, polygon_to_bbox)


class PointParsingError(ValueError):
    pass


class PointParserBase:
    @classmethod
    @abstractmethod
    def parse(cls, s: str) -> Point:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def serialize(cls, point: Point) -> str:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def match_candidate_pattern(cls) -> str:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def is_parser(cls, s: str) -> bool:
        raise NotImplementedError()


class HTMLPointParser(PointParserBase):
    """Centralised (de)serialisation engine for Point objects with time support."""

    _single_point_pattern = re.compile(r"\(\s*(?P<x>\d+)\s*,\s*(?P<y>\d+)\s*\)")
    _attr_t_pattern = re.compile(r"\bt\s*=\s*(?:\"(?P<dq>[^\"]+)\"|'(?P<sq>[^']+)'|(?P<nq>[^\s>]+))")

    @classmethod
    def serialize(cls, point: Point) -> str:
        body = point.serialize_coords()
        tag = point.__class__.point_name
        attr = f" t={point.t}" if point.t is not None else ""
        return f"{point.mention or ''} <{tag}{attr}> {body} </{tag}>".strip()

    @classmethod
    def _extract_t(cls, attr_str: str) -> float | None:
        t_match = cls._attr_t_pattern.search(attr_str)
        if not t_match:
            return None
        raw = t_match.group("dq") or t_match.group("sq") or t_match.group("nq")
        try:
            return float(raw)
        except ValueError:
            raise PointParsingError(f"Invalid time value t={raw}")

    @classmethod
    def _extract_points(cls, body: str) -> list[tuple[str, str]]:
        """Return all (x, y) pairs and fail if body contains anything else."""
        matches = cls._single_point_pattern.findall(body)
        return matches

    @classmethod
    def _parse_single_point(cls, s: str) -> SinglePoint:
        points = cls._extract_points(s)
        if len(points) != 1:
            raise PointParsingError(f"Expected exactly one point in '{s}'")
        x, y = map(int, points[0])
        return SinglePoint(x, y)

    @classmethod
    def _parse_bounding_box(cls, s: str) -> BoundingBox:
        points = cls._extract_points(s)
        if len(points) != 2:
            raise PointParsingError(f"Expected two points in '{s}'")
        (x1, y1), (x2, y2) = map(lambda p: (int(p[0]), int(p[1])), points)
        return BoundingBox(SinglePoint(x1, y1), SinglePoint(x2, y2))

    @classmethod
    def _parse_polygon(cls, s: str) -> Polygon:
        points = cls._extract_points(s)
        if not points:
            raise PointParsingError(f"No points found in '{s}'")
        return Polygon([SinglePoint(int(px), int(py)) for px, py in points])

    @classmethod
    def match_candidate_pattern(cls) -> str:
        #  (x , y)   —— same shape as _single_point_pattern, but without the named groups
        point = r"\(\s*\d+\s*,\s*\d+\s*\)"
        # full expression:   opening tag • attrs • body (≥1 point) • closing tag
        return rf"<(point(?:_box)?|polygon)(?:\s[^>]*)?>(\s*(?:{point}\s*)+)</\1>"

    @classmethod
    def parse(cls, s: str) -> Point:
        s = str(s).strip()
        # Identify opening tag and attributes
        m = re.match(r"<(?P<tag>\w+)(?P<attrs>(?:\s+[^>]+)?)>", s)
        if not m:
            raise PointParsingError(f"No opening tag found in '{s}'")
        tag = m.group("tag")
        attrs = m.group("attrs") or ""
        closing_idx = s.rfind(f"</{tag}>")
        if closing_idx == -1:
            raise PointParsingError(f"Missing closing tag for <{tag}> in '{s}'")
        body = s[m.end() : closing_idx].strip()

        if tag == SinglePoint.point_name:
            point = cls._parse_single_point(body)
        elif tag == BoundingBox.point_name:
            point = cls._parse_bounding_box(body)
        elif tag == Polygon.point_name:
            point = cls._parse_polygon(body)
        else:
            raise PointParsingError(f"{tag} is not supported")

        # attach time attribute if present
        t_val = cls._extract_t(attrs)
        if t_val is not None:
            point.t = t_val

        return point

    @classmethod
    def is_parser(cls, s: str) -> bool:
        m = re.match(r"<(?P<tag>\w+)>", s)
        if not m:
            # there should only be atleast one open tag to attempt parse
            return False

        # only valid if the tag does not
        return m.group("tag") in {SinglePoint.point_name, BoundingBox.point_name, Polygon.point_name}


class UnifiedPolygonParser(PointParserBase):
    """Treats all points like a polygon object"""

    @classmethod
    def serialize(cls, point: Point) -> str:
        if isinstance(point, SinglePoint):
            point = Polygon(hull=[point], t=point.t)
        elif isinstance(point, BoundingBox):
            point = bbox_to_polygon(point)
        return HTMLPointParser.serialize(point)

    @classmethod
    def parse(cls, s: str) -> Point:
        polygon = HTMLPointParser.parse(s)
        assert isinstance(polygon, Polygon)
        if len(polygon.hull) == 1:
            # single point
            single_point = polygon.hull[0]
            single_point.t = polygon.t
            return single_point
        elif len(polygon.hull) == 4:
            possible_box = polygon_to_bbox(polygon)
            if possible_box:
                # this is a valid bounding box
                possible_box.t = polygon.t
                return possible_box

        return polygon

    @classmethod
    def match_candidate_pattern(cls) -> str:
        point = r"\(\s*\d+\s*,\s*\d+\s*\)"
        return rf"<(polygon)(?:\s[^>]*)?>(\s*(?:{point}\s*)+)</\1>"

    @classmethod
    def is_parser(cls, s: str) -> bool:
        # only polygons are unified polygon parser
        m = re.match(r"<(?P<tag>\w+)>", s)
        if not m:
            # there should only be atleast one open tag to attempt parse
            return False

        # only valid if the tag does not
        return m.group("tag") == "polygon"


class XMLPointParser(PointParserBase):
    """Parse pointing data represented as <perc:pointer type="point|box|polygon" coords="..." t="...">...</perc:pointer>"""

    _tag = "perc:pointer"
    _single_point_pattern = re.compile(r"\(\s*(?P<x>\d+)\s*,\s*(?P<y>\d+)\s*\)")
    _attr_pattern = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


    @classmethod
    def serialize(cls, point: Point) -> str:
        coords = point.serialize_coords()

        # Get the XML type of the point.
        point_type = point.__class__.point_type()

        attrs = {"type": point_type, "coords": coords}
        if point.t is not None:
            attrs["t"] = point.t

        attrs_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
        if attrs_str:
            attrs_str = f" {attrs_str}"

        if point.mention:
            return f"<{cls._tag}{attrs_str}>{point.mention}</{cls._tag}>"
        else:
            return f"<{cls._tag}{attrs_str} />"

    @classmethod
    def _extract_attributes(cls, tag_content: str) -> dict[str, str]:
        """Extract attributes from XML tag content."""
        attrs = {}
        for match in cls._attr_pattern.finditer(tag_content):
            attrs[match.group(1)] = match.group(2)
        return attrs

    @classmethod
    def _extract_points(cls, body: str) -> list[tuple[str, str]]:
        """Return all (x, y) pairs and fail if body contains anything else."""
        matches = cls._single_point_pattern.findall(body)
        return matches

    @classmethod
    def _parse_single_point(cls, s: str) -> SinglePoint:
        points = cls._extract_points(s)
        if len(points) != 1:
            raise PointParsingError(f"Expected exactly one point in '{s}'")
        x, y = map(int, points[0])
        return SinglePoint(x, y)

    @classmethod
    def _parse_bounding_box(cls, s: str) -> BoundingBox:
        points = cls._extract_points(s)
        if len(points) != 2:
            raise PointParsingError(f"Expected two points in '{s}'")
        (x1, y1), (x2, y2) = map(lambda p: (int(p[0]), int(p[1])), points)
        return BoundingBox(SinglePoint(x1, y1), SinglePoint(x2, y2))

    @classmethod
    def _parse_polygon(cls, s: str) -> Polygon:
        points = cls._extract_points(s)
        if not points:
            raise PointParsingError(f"No points found in '{s}'")
        return Polygon([SinglePoint(int(px), int(py)) for px, py in points])

    @classmethod
    def match_candidate_pattern(cls) -> str:
        tag = cls._tag.replace(":", "\\:")  # Escape colon for regex
        return rf"<{tag}(?:\s[^>]*)?(?:\s*/>|>[^<]*</{tag}>)"

    @classmethod
    def parse(cls, s: str) -> Point:
        s = str(s).strip()
        tag = cls._tag.replace(":", "\\:")  # Escape colon for regex
        m = re.match(rf"<{tag}(?P<attrs>\s[^>]*)?(?:\s*/>|>(?P<body>[^<]*)</{tag}>)", s)

        if not m:
            raise PointParsingError(f"No opening tag found in '{s}'")
        attrs = cls._extract_attributes(m.group("attrs"))
        body = m.group("body") or ""

        if "type" not in attrs:
            raise PointParsingError(f"Missing type attribute in '{s}'")
        point_type = attrs["type"]
        coords = attrs.get("coords", "")
        t = attrs.get("t", None)

        if point_type == SinglePoint.point_type():
            point = cls._parse_single_point(coords)
        elif point_type == BoundingBox.point_type():
            point = cls._parse_bounding_box(coords)
        elif point_type == Polygon.point_type():
            point = cls._parse_polygon(coords)
        else:
            raise PointParsingError(f"{point_type} is not supported")

        # attach time attribute if present
        if t is not None:
            point.t = t

        if body:
            point.mention = body

        return point

    @classmethod
    def is_parser(cls, s: str) -> bool:
        # Only valid if the tag is <perc:pointer>
        return s.startswith(f"<{cls._tag}")

    @classmethod
    def convert_xml_to_html(cls, text: str) -> str:
        """
        Convert XML pointing data to HTML format within the given text.
        
        Args:
            text: Text containing XML pointing data (perc:pointer tags)
            
        Returns:
            Text with XML pointing data converted to HTML format
        """
        # Use the match_candidate_pattern to find and replace all XML tags
        pattern = cls.match_candidate_pattern()

        def replace_xml_with_html(match):
            xml_tag = match.group(0)
            try:
                # Parse the XML tag to get the Point object
                point = cls.parse(xml_tag)
                # Serialize it using HTMLPointParser
                html_tag = HTMLPointParser.serialize(point)
                return html_tag
            except PointParsingError:
                # If parsing fails, return the original tag unchanged
                return xml_tag

        # Replace all XML tags with HTML equivalents
        return re.sub(pattern, replace_xml_with_html, text, flags=re.DOTALL)

    @classmethod
    def remove_pointing_tags(cls, text: str) -> str:
        """
        Removes perc:pointer tags from the text, returning text response only.
        
        Args:
            text: Text containing XML pointing data (perc:pointer tags)
            
        Returns:
            Text with XML pointing data removed
        """

        return re.sub(r"(?:<perc:pointer)[^>]*>|(?:</perc:pointer>)", "", text, flags=re.DOTALL)


SUPPORTED_PARSERS: list[PointParserBase] = [HTMLPointParser, UnifiedPolygonParser]


def parse(input: str, parsers: list[PointParserBase] | None) -> Point:
    """
    Parse the input string across the supported parsing formats, takes an ordered list of
    parsers to try parsing.
    """
    if not parsers:
        parsers = SUPPORTED_PARSERS
    for parser in parsers:
        try:
            return parser.parse(input)
        except PointParsingError:
            continue

    raise PointParsingError(f"Unable to match {input} to {parsers}")


@dataclass
class ExtractedPoints:
    original_text: str
    deliexicalized_text: str
    id_to_point: dict[str, Point]

    def replace_points(self, replacement_fn: Callable[[str, Point], str]) -> str:
        """
        Replace all point tokens with custom content using a replacement function.
        """
        result = self.deliexicalized_text
        
        # Process point IDs in reverse order to avoid substring matching issues
        # (e.g., POINT_10 should be processed before POINT_1 to avoid corruption)
        for point_id in reversed(list(self.id_to_point.keys())):
            point = self.id_to_point[point_id]
            replacement = replacement_fn(point_id, point)
            result = result.replace(point_id, replacement)
        
        return result

def extract_points(text_string: str, parser: PointParserBase) -> ExtractedPoints:
    mapping = {}

    # Counter to assign unique IDs
    count = 0

    # Function to be used in re.sub replacement
    def replace(match):
        proposed_point: str = match.group(0)
        try:
            point = parser.parse(proposed_point)
        except PointParsingError:
            # this is not a point because it failed validation
            return proposed_point

        nonlocal count
        token = f"POINT_{count}"
        # Save the entire matched point string (group 0)
        mapping[token] = point
        count += 1
        return token

    # Replace all occurrences using the above function
    deliexicalized_text = re.sub(parser.match_candidate_pattern(), replace, text_string, flags=re.DOTALL)

    # Return the original text, the deliexicalized text, and the mapping dictionary
    return ExtractedPoints(
        original_text=text_string,
        deliexicalized_text=deliexicalized_text,
        id_to_point=mapping,
    )

def convert_pointing_format(text: str, source_parser_cls: type[PointParserBase], target_parser_cls: type[PointParserBase]) -> str:
    """
    Convert pointing format from one parser to another.
    For example, let's say you have response text with pointing format in XML format:
    '<perc:pointer type="point" coords="(100,200)">submit button</perc:pointer>'
    You can convert it to HTML format like:
    'submit button <point> (100,200) </point>'
    by using the following code:
        convert_pointing_format(text, XMLPointParser, HTMLPointParser)
    """
    extracted_points = extract_points(text, source_parser_cls)
    return extracted_points.replace_points(lambda _, point: target_parser_cls.serialize(point))
