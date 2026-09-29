"""Load Phase-1 portfolio JSON facts from the configured data directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_FILES = (
    "profile.json",
    "projects.json",
    "experience.json",
    "education.json",
    "courses.json",
)


class ContentLoadError(RuntimeError):
    """Raised when required content files are missing or invalid."""


class ContentStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._cache: dict[str, Any] = {}

    def load_all(self) -> None:
        """Validate and cache all required JSON files. Fail loudly on errors."""
        if not self.data_dir.is_dir():
            raise ContentLoadError(f"DATA_DIR does not exist or is not a directory: {self.data_dir}")

        loaded: dict[str, Any] = {}
        for name in REQUIRED_FILES:
            loaded[name] = self._read_json(name)
        self._validate_shapes(loaded)
        self._cache = loaded

    def _read_json(self, filename: str) -> Any:
        path = self.data_dir / filename
        if not path.is_file():
            raise ContentLoadError(f"Required content file missing: {path}")
        try:
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError as exc:
            raise ContentLoadError(f"Corrupt JSON in {path}: {exc}") from exc

    @staticmethod
    def _validate_shapes(loaded: dict[str, Any]) -> None:
        profile = loaded["profile.json"]
        if not isinstance(profile, dict) or "name" not in profile:
            raise ContentLoadError("profile.json must be an object with at least a name field")

        projects_doc = loaded["projects.json"]
        if not isinstance(projects_doc, dict) or not isinstance(projects_doc.get("projects"), list):
            raise ContentLoadError("projects.json must contain a projects array")
        if len(projects_doc["projects"]) == 0:
            raise ContentLoadError("projects.json projects array must not be empty")

        for name in ("experience.json", "education.json", "courses.json"):
            doc = loaded[name]
            if not isinstance(doc, dict) or "items" not in doc:
                raise ContentLoadError(f"{name} must be an object with an items array")
            if not isinstance(doc["items"], list):
                raise ContentLoadError(f"{name} items must be a list")

    def get_profile(self) -> dict[str, Any]:
        return self._cache["profile.json"]

    def get_projects(self) -> list[dict[str, Any]]:
        return list(self._cache["projects.json"]["projects"])

    def get_project(self, slug: str) -> dict[str, Any] | None:
        for project in self.get_projects():
            if project.get("slug") == slug:
                return project
        return None

    def get_experience(self) -> dict[str, Any]:
        return self._cache["experience.json"]

    def get_education(self) -> dict[str, Any]:
        return self._cache["education.json"]

    def get_courses(self) -> dict[str, Any]:
        return self._cache["courses.json"]
