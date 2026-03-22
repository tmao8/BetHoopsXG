import predict
import json
import os

stats = ["Points", "Rebounds", "Assists"]
os.makedirs("ui/public/api", exist_ok=True)

for stat in stats:
    print(f"Generating predictions for {stat}...")
    df = predict.predict(stat)
    
    # Save directly into the React public folder
    data = {"status": "success", "data": df.to_dict(orient="records")}
    with open(f"ui/public/api/latest_{stat}.json", "w") as f:
        json.dump(data, f)

print("All predictions generated and saved!")
