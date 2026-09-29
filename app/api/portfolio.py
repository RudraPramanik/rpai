"""Read-only portfolio API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas.portfolio import (
    HealthResponse,
    ItemsResponse,
    Profile,
    Project,
    ProjectsResponse,
)
from app.services.content import ContentStore

router = APIRouter()


def _store(request: Request) -> ContentStore:
    return request.app.state.content_store


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/api/profile", response_model=Profile, tags=["portfolio"])
def get_profile(request: Request) -> Profile:
    return Profile.model_validate(_store(request).get_profile())


@router.get("/api/projects", response_model=ProjectsResponse, tags=["portfolio"])
def list_projects(request: Request) -> ProjectsResponse:
    projects = [Project.model_validate(item) for item in _store(request).get_projects()]
    return ProjectsResponse(projects=projects)


@router.get("/api/projects/{slug}", response_model=Project, tags=["portfolio"])
def get_project(slug: str, request: Request) -> Project:
    project = _store(request).get_project(slug)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {slug}")
    return Project.model_validate(project)


@router.get("/api/experience", response_model=ItemsResponse, tags=["portfolio"])
def get_experience(request: Request) -> ItemsResponse:
    return ItemsResponse.model_validate(_store(request).get_experience())


@router.get("/api/education", response_model=ItemsResponse, tags=["portfolio"])
def get_education(request: Request) -> ItemsResponse:
    return ItemsResponse.model_validate(_store(request).get_education())


@router.get("/api/courses", response_model=ItemsResponse, tags=["portfolio"])
def get_courses(request: Request) -> ItemsResponse:
    return ItemsResponse.model_validate(_store(request).get_courses())
