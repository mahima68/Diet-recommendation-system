# Diet Recommendation System

A Python portfolio project that retrieves complete breakfast, lunch, dinner, and snack plans from a supplied dataset. Includes an interactive terminal interface, notebook, reusable recommendation module, portable saved artifact, and automated tests.

## Quick start

Requires Python 3.10 or newer. Run commands inside the project folder:

```sh
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows instead:
# .venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

For a reproducible example and CSV export:

```sh
python app.py --age 42 --weight 103 --activity "Moderately Active" --preference Vegan --conditions Acne --calories 2000 --tolerance 0.1 --export recommendations.csv
```

Open `diet_recommendation_model.ipynb` with Jupyter for an example, data audit, filtering, evaluation, and artifact export:

```sh
jupyter notebook diet_recommendation_model.ipynb
```

## Recommendation behavior

1. Validate the adult profile, activity level, dietary category, and condition labels.
2. Filter by the exact dietary category in the dataset. Preferences are never silently relaxed.
3. Exclude obvious animal-ingredient conflicts using a conservative text screen (including common meat, fish, dairy, and egg terms; explicitly named plant alternatives are allowed). This cannot certify ingredients.
4. Apply optional calorie tolerance and explicit nutrient minimum/maximum values.
5. Rank eligible rows using cosine similarity over standardized age, weight, activity, and condition indicators. Preprocessing is fitted once when the recommender is constructed.
6. If a calorie target is supplied, blend profile similarity (70%) with calorie proximity (30%). These weights are a transparent heuristic, not a validated clinical rule.
7. Remove repeated combinations of meals and return up to the requested number of plans.

An empty result means no plans passed the filters. Scores are ranking values, not probabilities or medical confidence. A calorie target must be supplied explicitly; the system does not estimate nutritional needs from age and weight.

## Python API

```python
from recommender import DietRecommender, Profile

model = DietRecommender.from_csv()
profile = Profile(age=42, weight=103, activity="Moderately Active",
                  preference="Vegan", conditions=("Acne",), calorie_target=2000)
plans = model.recommend(profile, top_k=5, calorie_tolerance=0.1)
# Optional: nutrient_max={"Sugar": 150}, nutrient_min={"Protein": 60}
# Limits use the dataset's raw units; verify units before interpreting them.
model.save("diet_recommendation_model.json")
loaded = DietRecommender.load("diet_recommendation_model.json")
```

The JSON artifact includes the dataset, feature names, means, and scales. Loading validates preprocessing consistency and reconstructs the recommender without executing pickle content. The original `.pkl` is superseded and is not needed.

## Dataset limitations

The supplied CSV contains 1,698 rows. Its original source, license, nutritional units, serving sizes, and whether nutrient values describe individual meals or whole plans have not been documented. The CSV is preserved without invented provenance or units. Dietary filtering enforces category labels and excludes recognizable ingredient conflicts; ambiguous or unlisted ingredients can still pass: meal ingredients and category labels have not been independently verified. Disease labels indicate dataset matching, not evidence that a plan treats or is appropriate for a condition. No allergy filtering is implemented because the dataset has no ingredient/allergen metadata.

This is an educational retrieval demo. Verify the dataset and obtain appropriate nutritional review before using it for real dietary decisions. Nutrient limits are explicitly supplied by the caller; the system does not infer medical restrictions.

## Validation

```sh
python -m unittest -v
```

Tests cover the original vegan-profile regression, every dietary category across different profiles, calorie and nutrient constraints, empty results, invalid input, stable ordering, unique meal combinations, constant-feature behavior, dataset validation, and saved-artifact round trips. These checks validate software behavior, not nutritional quality.

## Files

- `recommender.py`: reusable validated recommendation logic.
- `app.py`: interactive terminal interface and CSV export.
- `diet_recommendation_model.ipynb`: updated walkthrough and checks.
- `diet_recommendation_model.json`: reusable dataset and preprocessing artifact.
- `Food_and_Nutrition__.csv`: original dataset.
- `original_notebook.ipynb`: original notebook retained for reference.
- `test_recommender.py`: automated regression tests.
