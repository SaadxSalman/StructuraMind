"""IFC / BIM metadata parser.

Extracts spatial and material metadata from Industry Foundation Classes (IFC)
files using IfcOpenShell. When IfcOpenShell is not installed (common on some
platforms), the parser degrades gracefully to a stub so the rest of the
application keeps working.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger("structuramind.ingestion.ifc_parser")


@dataclass
class IFCElement:
    """A single extracted IFC element (wall, beam, column, slab, ...)."""

    element_id: str
    ifc_guid: str
    ifc_type: str  # e.g. IfcWall, IfcBeam, IfcColumn, IfcSlab
    name: Optional[str]
    description: Optional[str]
    level: Optional[str]  # building storey
    material: Optional[str]
    quantity: Optional[dict]
    bbox: Optional[List[float]]  # [xmin, ymin, zmin, xmax, ymax, zmax]
    properties: Dict[str, object] = field(default_factory=dict)


@dataclass
class ParsedIFCModel:
    """Result of parsing an IFC file."""

    document_id: str
    filename: str
    elements: List[IFCElement] = field(default_factory=list)
    element_count: int = 0
    metadata: dict = field(default_factory=dict)


class IFCParser:
    """Parse an IFC file into structured element records."""

    def __init__(self) -> None:
        self._ifcopenshell = self._load_ifcopenshell()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def parse_file(self, file_path: str, document_id: Optional[str] = None) -> ParsedIFCModel:
        """Parse an IFC file on disk."""
        if document_id is None:
            document_id = str(uuid.uuid4())
        filename = file_path.split("/")[-1].split("\\")[-1]
        return self._parse(document_id, filename, file_path=file_path)

    def parse_bytes(self, file_bytes: bytes, filename: str, document_id: Optional[str] = None) -> ParsedIFCModel:
        """Parse IFC bytes (writes to a temp file for IfcOpenShell)."""
        if document_id is None:
            document_id = str(uuid.uuid4())
        import tempfile
        import os

        suffix = ".ifc"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            return self._parse(document_id, filename, file_path=tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_ifcopenshell():
        try:
            import ifcopenshell  # type: ignore
            logger.info("ifcopenshell loaded successfully")
            return ifcopenshell
        except Exception as exc:  # noqa: BLE001 - graceful degradation
            logger.warning(
                "ifcopenshell not available (%s). IFC parsing will return a stub.", exc
            )
            return None

    def _parse(self, document_id: str, filename: str, file_path: str) -> ParsedIFCModel:
        if self._ifcopenshell is None:
            logger.warning("IFC stub: returning empty model for '%s'", filename)
            return ParsedIFCModel(
                document_id=document_id,
                filename=filename,
                elements=[],
                element_count=0,
                metadata={"parser": "stub", "ifcopenshell": False},
            )

        try:
            model = self._ifcopenshell.open(file_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to open IFC file '%s': %s", filename, exc)
            return ParsedIFCModel(
                document_id=document_id,
                filename=filename,
                elements=[],
                element_count=0,
                metadata={"parser": "ifcopenshell", "error": str(exc)},
            )

        elements: List[IFCElement] = []
        storeys: Dict[str, str] = {}
        for storey in model.by_type("IfcBuildingStorey"):
            storeys[storey.GlobalId] = storey.Name or storey.GlobalId

        target_types = [
            "IfcWall", "IfcBeam", "IfcColumn", "IfcSlab", "IfcFooting",
            "IfcPile", "IfcMember", "IfcPlate", "IfcStructuralMember",
        ]
        seen_guids = set()
        for ifc_type in target_types:
            for entity in model.by_type(ifc_type):
                guid = getattr(entity, "GlobalId", None) or str(uuid.uuid4())
                if guid in seen_guids:
                    continue
                seen_guids.add(guid)
                element = self._extract_element(entity, storeys)
                if element is not None:
                    elements.append(element)

        logger.info("Parsed IFC '%s': %d elements extracted", filename, len(elements))
        return ParsedIFCModel(
            document_id=document_id,
            filename=filename,
            elements=elements,
            element_count=len(elements),
            metadata={"parser": "ifcopenshell", "ifcopenshell": True},
        )


    def _extract_element(self, entity, storeys: Dict[str, str]) -> Optional[IFCElement]:
        """Extract a single IFC element into an :class:`IFCElement`."""
        try:
            guid = getattr(entity, "GlobalId", None) or str(uuid.uuid4())
            name = getattr(entity, "Name", None)
            description = getattr(entity, "Description", None)
            ifc_type = entity.is_a() if hasattr(entity, "is_a") else "IfcElement"

            # Resolve level (storey) via containment relationship.
            level = None
            try:
                for rel in getattr(entity, "ContainedInStructure", []) or []:
                    relating = getattr(rel, "RelatingStructure", None)
                    if relating is not None and relating.is_a() == "IfcBuildingStorey":
                        level = storeys.get(relating.GlobalId, relating.GlobalId)
                        break
            except Exception:
                pass

            # Material.
            material = None
            try:
                for rel in getattr(entity, "HasAssociations", []) or []:
                    if rel.is_a() == "IfcRelAssociatesMaterial":
                        mat = getattr(rel, "RelatingMaterial", None)
                        if mat is not None:
                            material = getattr(mat, "Name", None) or getattr(mat, "Description", None)
                            break
            except Exception:
                pass

            # Bounding box + quantities (best effort).
            bbox = None
            quantity: Optional[dict] = None
            properties: Dict[str, object] = {}
            try:
                rep = getattr(entity, "Representation", None)
                if rep is not None:
                    bbox = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            except Exception:
                pass

            # Property sets.
            try:
                for rel in getattr(entity, "IsDefinedBy", []) or []:
                    if rel.is_a() == "IfcRelDefinesByProperties":
                        prop_set = getattr(rel, "RelatingPropertySet", None)
                        if prop_set is not None:
                            pset_name = getattr(prop_set, "Name", "properties")
                            props = getattr(prop_set, "HasProperties", []) or []
                            pset_vals = {}
                            for prop in props:
                                pname = getattr(prop, "Name", None)
                                pval = self._prop_value(prop)
                                if pname is not None:
                                    pset_vals[pname] = pval
                            properties[pset_name] = pset_vals
            except Exception:
                pass

            return IFCElement(
                element_id=str(uuid.uuid4()),
                ifc_guid=guid,
                ifc_type=ifc_type,
                name=name,
                description=description,
                level=level,
                material=material,
                quantity=quantity,
                bbox=bbox,
                properties=properties,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to extract IFC element: %s", exc)
            return None


    @staticmethod
    def _prop_value(prop) -> object:
        """Extract a scalar value from an IFC property object."""
        try:
            if hasattr(prop, "NominalValue") and prop.NominalValue is not None:
                return prop.NominalValue.wrappedValue
        except Exception:
            pass
        return None

