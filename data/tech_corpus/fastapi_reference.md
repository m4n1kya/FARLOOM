# FastAPI — Technical Reference

## Overview

FastAPI is a modern, fast Python web framework for building APIs with Python type hints.
It is built on Starlette for the web layer and Pydantic for data validation.
FastAPI automatically generates OpenAPI (Swagger) documentation.

Key characteristics:
- Based on Python 3.11+ type hints
- Automatic request/response validation via Pydantic
- Async-first: supports `async def` and `def` endpoints
- Dependency injection system
- Auto-generated docs at `/docs` (Swagger UI) and `/redoc`

## Installation

```bash
pip install "fastapi[standard]"    # includes uvicorn
pip install fastapi uvicorn httpx
```

## Minimal Application

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Hello, World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None) -> dict:
    return {"item_id": item_id, "q": q}
```

Run with: `uvicorn main:app --reload`

## Path Parameters and Query Parameters

```python
from fastapi import Query
from typing import Annotated

@app.get("/search")
def search(
    q: Annotated[str, Query(min_length=1, description="Search term")],
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> dict:
    return {"q": q, "limit": limit}
```

## Request Bodies with Pydantic

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = False

@app.post("/items/")
def create_item(item: Item) -> Item:
    return item
```

FastAPI validates the request body against the Pydantic model and returns 422
if validation fails.

## Response Models

```python
class ItemResponse(BaseModel):
    id: int
    name: str

@app.get("/items/{item_id}", response_model=ItemResponse)
def get_item(item_id: int) -> ItemResponse:
    return ItemResponse(id=item_id, name="Widget")
```

The `response_model` parameter filters the output to only the declared fields.

## Dependency Injection

```python
from fastapi import Depends

def get_db() -> Database:
    db = Database()
    try:
        yield db
    finally:
        db.close()

DBDep = Annotated[Database, Depends(get_db)]

@app.get("/users")
def list_users(db: DBDep) -> list[User]:
    return db.all_users()
```

`Depends` supports: regular functions, async functions, class-based
dependencies, and nested dependencies.

## Application Lifecycle (lifespan)

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # startup
    app.state.model = load_model()
    yield
    # shutdown
    app.state.model.close()

app = FastAPI(lifespan=lifespan)
```

`lifespan` REPLACES the deprecated `on_event("startup")` handlers.

## HTTP Status Codes and Exceptions

```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
def get_item(item_id: int) -> Item:
    item = db.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item
```

Common codes: 200 OK, 201 Created, 400 Bad Request, 404 Not Found, 422 Unprocessable Entity,
500 Internal Server Error.

## Middleware

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

CORS middleware is REQUIRED when the frontend is served from a different origin than the API.

## Static Files

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root() -> FileResponse:
    return FileResponse("static/index.html")
```

## TestClient

```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
```

`TestClient` wraps `httpx.Client` and runs the ASGI app in-process.
No server is started; tests run without network access.

## Performance Notes

- FastAPI + uvicorn achieves ~50,000 requests/second on simple endpoints.
- Use `async def` endpoints for I/O-bound operations.
- Use `def` endpoints for CPU-bound operations (FastAPI runs them in a thread pool).
- Enable `--workers N` in production: `uvicorn app:app --workers 4`

## Versioning and Compatibility

FastAPI follows semantic versioning. Version 0.100+ introduces breaking changes
from the 0.9x series including the removal of `on_event` in favor of `lifespan`.
Pydantic v2 is required by FastAPI >= 0.100. Pydantic v1 is not compatible with
FastAPI >= 0.100 without the `pydantic.v1` compatibility shim.
