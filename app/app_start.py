from app.main import app as main_app
from frontend.main import app as frontend_app

if __name__ == "__main__":
    import uvicorn
    # uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    # uvicorn frontend.main:app --host 0.0.0.0 --port 8080 --reload
    uvicorn.run(main_app, host="0.0.0.0", port=8000, reload=True)
    uvicorn.run(frontend_app, host="0.0.0.0", port=8080, reload=True)
