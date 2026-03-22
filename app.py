from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/predictions")
def get_predictions(stat: str = Query(default="Points", description="Stat type to predict: Points, Rebounds, Assists")):
    # Map stat to shorthand used in code
    stat_mapping = {
        "Points": "PTS",
        "Rebounds": "REB",
        "Assists": "AST"
    }
    target_stat = stat_mapping.get(stat, "PTS")
    
    # Run prediction
    try:
        df = predict.predict(target_stat)
        # Convert to dictionary for JSON response
        return {"status": "success", "data": df.to_dict(orient="records")}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
