from backend.main import app

if __name__ == "__main__":
    import uvicorn
    # uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
