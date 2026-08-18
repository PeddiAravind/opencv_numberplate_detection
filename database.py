import pandas as pd
from rapidfuzz import process

# Load database
df = pd.read_csv("database/vehicles.csv")


def search_plate(plate):

    plate = plate.replace(" ", "").upper()

    # Exact Matches
    result = df[df["plate_number"] == plate]

    if not result.empty:
        return result.iloc[0]

    # Fuzzy Match
    choices = df["plate_number"].tolist()

    best_match = process.extractOne(plate, choices)

    if best_match is None:
        return None

    matched_plate, score, index = best_match

    if score >= 80:
        print(f"\nClosest Match : {matched_plate}")
        print(f"Similarity    : {score:.1f}%")

        return df.iloc[index]

    return None

if __name__ == "__main__":

    test = "TSO7JS967O"

    result = search_plate(test)

    print(result)
