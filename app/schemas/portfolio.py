"""Pydantic response models aligned to Phase-1 JSON shapes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProfileLinks(BaseModel):
    model_config = ConfigDict(extra="allow")

    github: str | None = None
    linkedin: str | None = None
    instagram: str | None = None


class Profile(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    headline: str | None = None
    tagline: str | None = None
    location: str | None = None
    email: str | None = None
    links: ProfileLinks | dict[str, Any] = Field(default_factory=dict)
    techStack: list[str] = Field(default_factory=list)
    yearsOfExperience: int | None = None
    resumeUrl: str | None = None


class Project(BaseModel):
    model_config = ConfigDict(extra="allow")

    slug: str
    title: str
    repositoryUrl: str | None = None
    demoUrl: str | None = None
    imageKey: str | None = None
    tabs: list[str] = Field(default_factory=list)
    category: str | None = None
    projectType: str | None = None
    summary: str | None = None
    techStacks: list[str] = Field(default_factory=list)


class ProjectsResponse(BaseModel):
    projects: list[Project]


class ItemsResponse(BaseModel):
    items: list[Any] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
