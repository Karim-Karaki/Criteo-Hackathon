# Run this locally to see your exact category names
import pandas as pd
import json

with open(r"C:\Users\karim\Desktop\Data Science Projects\Vast AI\Data\level2_categories.json", "r") as f:
    mapping = json.load(f)

taxonomy = pd.DataFrame(mapping).rename(columns={"new_id": "category_id", "google_id": "level_2_id"})
print(taxonomy[["category_id", "category_name"]].sort_values("category_id").to_string())