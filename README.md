# backend :

cd backend
python -m venv venv
./venv/Scripts/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 9000

# frontend

cd frontend
npm i
npm start

# infra (optional)

To start Postgres + Neo4j + Qdrant locally:

```
cd infra
docker compose -f docker-compose.kag.yml up -d
```
