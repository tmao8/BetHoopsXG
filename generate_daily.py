import predict
import evaluate_performance
import json
import os
import time

stats = ["Points", "Rebounds", "Assists"]
os.makedirs("ui/public/api", exist_ok=True)

# Grade yesterday's predictions BEFORE overwriting with today's
evaluate_performance.run()

for stat in stats:
    print(f"Generating predictions for {stat}...")
    df = predict.predict(stat)
    
    # Save directly into the React public folder
    data = {"status": "success", "data": df.to_dict(orient="records")}
    with open(f"ui/public/api/latest_{stat}.json", "w") as f:
        json.dump(data, f)
        
    time.sleep(4)

print("All predictions generated and saved!")
