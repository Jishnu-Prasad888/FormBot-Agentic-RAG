# backend :

cd backend
python -m venv venv
./venv/Scripts/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# frontend

cd frontend
npm i
npm start
