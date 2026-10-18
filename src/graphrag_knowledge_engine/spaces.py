"""Knowledge Spaces: Isolated multi-tenant environments for FARLOOM.

Each Space has its own completely isolated storage:
- SQLite document, triple, and mention store
- ChromaDB vector collection
- In-memory NetworkX knowledge graph
- BM25 sparse index
- Uploaded source documents
"""

from __future__ import annotations

import json
import re
import shutil
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graphrag_knowledge_engine.config import EngineConfig
from graphrag_knowledge_engine.engine import Engine, create_engine


def slugify(text: str) -> str:
    """Normalize a space name into a clean, filesystem-safe identifier."""
    slug = re.sub(r"[^\w\s-]", "", text.strip().lower())
    slug = re.sub(r"[-\s]+", "-", slug).strip("-_")
    return slug or "space"


@dataclass
class SpaceMetadata:
    id: str
    name: str
    description: str
    created_at: str
    preset: str = "blank"  # blank, tech_docs, research, api_specs
    chunk_size: int = 800
    doc_count: int = 0
    chunk_count: int = 0
    entity_count: int = 0
    edge_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpaceManager:
    """Manages multiple isolated FARLOOM knowledge spaces."""

    def __init__(self, base_data_dir: Path, base_config: EngineConfig) -> None:
        self.base_data_dir = base_data_dir
        self.base_config = base_config
        self.spaces_root = base_data_dir / "spaces"
        self.spaces_root.mkdir(parents=True, exist_ok=True)
        self._engines: dict[str, Engine] = {}
        # Ensure default space exists
        self._ensure_space_dir("default", "Default Space", "System baseline knowledge space")

    def _ensure_space_dir(
        self,
        space_id: str,
        name: str,
        description: str = "",
        preset: str = "blank",
        chunk_size: int = 800,
    ) -> Path:
        space_dir = self.spaces_root / space_id
        space_dir.mkdir(parents=True, exist_ok=True)
        (space_dir / "uploads").mkdir(parents=True, exist_ok=True)
        meta_file = space_dir / "meta.json"
        if not meta_file.exists():
            meta = SpaceMetadata(
                id=space_id,
                name=name,
                description=description,
                preset=preset,
                chunk_size=chunk_size,
                created_at=datetime.now(UTC).isoformat(),
            )
            meta_file.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
        return space_dir

    def get_space_dir(self, space_id: str) -> Path:
        clean_id = slugify(space_id)
        space_dir = self.spaces_root / clean_id
        if not space_dir.exists():
            raise KeyError(f"Space '{space_id}' does not exist.")
        return space_dir

    def get_space_meta(self, space_id: str) -> SpaceMetadata:
        space_dir = self.get_space_dir(space_id)
        meta_file = space_dir / "meta.json"
        if meta_file.exists():
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                return SpaceMetadata(**data)
            except Exception:
                pass
        return SpaceMetadata(
            id=space_dir.name,
            name=space_dir.name.capitalize(),
            description="",
            created_at=datetime.now(UTC).isoformat(),
        )

    def get_engine(self, space_id: str = "default") -> Engine:
        clean_id = slugify(space_id)
        if clean_id in self._engines:
            return self._engines[clean_id]

        space_dir = self.get_space_dir(clean_id)
        meta = self.get_space_meta(clean_id)
        # Create a dedicated EngineConfig pointing to this space's directory
        space_config = self.base_config.model_copy(
            update={"data_dir": space_dir, "chunk_size": meta.chunk_size}
        )
        engine = create_engine(space_config, space_id=clean_id)
        self._engines[clean_id] = engine
        return engine

    def set_engine(self, space_id: str, engine: Engine) -> None:
        """Inject an existing engine (useful for tests or predefined fixtures)."""
        clean_id = slugify(space_id)
        self._ensure_space_dir(clean_id, clean_id.capitalize())
        self._engines[clean_id] = engine

    def list_spaces(self) -> list[SpaceMetadata]:
        spaces: list[SpaceMetadata] = []
        for path in sorted(self.spaces_root.iterdir()):
            if not path.is_dir():
                continue
            meta_file = path / "meta.json"
            if meta_file.exists():
                try:
                    data = json.loads(meta_file.read_text(encoding="utf-8"))
                    meta = SpaceMetadata(**data)
                except Exception:
                    meta = SpaceMetadata(
                        id=path.name,
                        name=path.name.capitalize(),
                        description="",
                        created_at=datetime.now(UTC).isoformat(),
                    )
            else:
                meta = SpaceMetadata(
                    id=path.name,
                    name=path.name.capitalize(),
                    description="",
                    created_at=datetime.now(UTC).isoformat(),
                )

            # Refresh live counts if engine is loaded or from store
            if path.name in self._engines:
                eng = self._engines[path.name]
                meta.doc_count = len(eng.store.list_documents())
                meta.chunk_count = len(eng.store.all_chunks())
                meta.entity_count = eng.graph.num_nodes
                meta.edge_count = eng.graph.num_edges
            elif (path / "store.sqlite").exists():
                with suppress(Exception):
                    eng = self.get_engine(path.name)
                    meta.doc_count = len(eng.store.list_documents())
                    meta.chunk_count = len(eng.store.all_chunks())
                    meta.entity_count = eng.graph.num_nodes
                    meta.edge_count = eng.graph.num_edges

            spaces.append(meta)
        return spaces

    def create_space(
        self,
        name: str,
        description: str = "",
        preset: str = "blank",
        chunk_size: int = 800,
    ) -> SpaceMetadata:
        clean_id = slugify(name)
        space_dir = self.spaces_root / clean_id
        if space_dir.exists():
            meta_file = space_dir / "meta.json"
            if meta_file.exists():
                return SpaceMetadata(**json.loads(meta_file.read_text(encoding="utf-8")))

        self._ensure_space_dir(clean_id, name, description, preset, chunk_size)
        meta = SpaceMetadata(
            id=clean_id,
            name=name,
            description=description,
            preset=preset,
            chunk_size=chunk_size,
            created_at=datetime.now(UTC).isoformat(),
        )
        (space_dir / "meta.json").write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
        return meta

    def delete_space(self, space_id: str) -> bool:
        clean_id = slugify(space_id)
        if clean_id == "default":
            raise ValueError("The 'default' space cannot be deleted.")

        space_dir = self.spaces_root / clean_id
        if not space_dir.exists():
            return False

        # Close running engine if cached
        if clean_id in self._engines:
            eng = self._engines.pop(clean_id)
            with suppress(Exception):
                eng.close()

        shutil.rmtree(space_dir, ignore_errors=True)
        return True

    def close(self) -> None:
        """Close all cached space engines and database drivers."""
        for eng in self._engines.values():
            with suppress(Exception):
                eng.close()
        self._engines.clear()
        from graphrag_knowledge_engine.store.neo4j_store import close_all_drivers

        with suppress(Exception):
            close_all_drivers()

