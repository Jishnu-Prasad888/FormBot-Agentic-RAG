from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.database import models


class KAGRepository:
    async def upsert_form_version(
        self,
        db: AsyncSession,
        form_id: str,
        version: str,
        data: dict,
    ) -> models.FormVersion:
        existing = await db.execute(
            select(models.FormVersion).where(
                models.FormVersion.form_id == form_id,
                models.FormVersion.version == version,
            )
        )
        form_version = existing.scalar_one_or_none()
        if form_version is None:
            form_version = models.FormVersion(id=data.get("id"), form_id=form_id, version=version)
            db.add(form_version)
        for k, v in data.items():
            if v is not None:
                setattr(form_version, k, v)
        await db.commit()
        await db.refresh(form_version)
        return form_version

    async def bulk_upsert_fields(self, db: AsyncSession, fields: list[dict]) -> int:
        if not fields:
            return 0
        objects = []
        for f in fields:
            obj = models.Field(**f)
            objects.append(obj)
        db.add_all(objects)
        await db.commit()
        return len(objects)

    async def bulk_upsert_requirements(self, db: AsyncSession, reqs: list[dict]) -> int:
        if not reqs:
            return 0
        objects = [models.Requirement(**r) for r in reqs]
        db.add_all(objects)
        await db.commit()
        return len(objects)

    async def bulk_upsert_form_requirements(self, db: AsyncSession, links: list[dict]) -> int:
        if not links:
            return 0
        db.add_all([models.FormRequirement(**l) for l in links])
        await db.commit()
        return len(links)

    async def bulk_upsert_form_regulations(self, db: AsyncSession, links: list[dict]) -> int:
        if not links:
            return 0
        db.add_all([models.FormRegulation(**l) for l in links])
        await db.commit()
        return len(links)

    async def bulk_upsert_field_dependencies(self, db: AsyncSession, deps: list[dict]) -> int:
        if not deps:
            return 0
        db.add_all([models.FieldDependency(**d) for d in deps])
        await db.commit()
        return len(deps)

    async def upsert_regulation(self, db: AsyncSession, data: dict) -> models.Regulation:
        existing = await db.execute(
            select(models.Regulation).where(models.Regulation.id == data.get("id"))
        )
        regulation = existing.scalar_one_or_none()
        if regulation is None:
            regulation = models.Regulation(**data)
            db.add(regulation)
        else:
            for k, v in data.items():
                if v is not None:
                    setattr(regulation, k, v)
        await db.commit()
        await db.refresh(regulation)
        return regulation

    async def delete_form_version(self, db: AsyncSession, form_version_id: str) -> None:
        await db.execute(delete(models.FormVersion).where(models.FormVersion.id == form_version_id))
        await db.commit()


kag_repo = KAGRepository()
