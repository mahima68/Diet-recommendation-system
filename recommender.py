"""Diet-plan retrieval demo. Dataset labels are not clinical evidence."""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
import numpy as np
import pandas as pd

ACTIVITY = {'Sedentary': 0, 'Lightly Active': 1, 'Moderately Active': 2,
            'Very Active': 3, 'Extremely Active': 4}
PREFERENCES = ('Omnivore', 'Pescatarian', 'Vegan', 'Vegetarian')
MEALS = ['Breakfast Suggestion', 'Lunch Suggestion', 'Dinner Suggestion', 'Snack Suggestion']
NUTRIENTS = ['Protein', 'Sugar', 'Sodium', 'Calories', 'Carbohydrates', 'Fiber', 'Fat']
MEAT = r"\b(chicken|beef|pork|steak|turkey|bacon|ham|lamb|sausage)\b"
FISH = r"\b(fish|salmon|tuna|shrimp|prawn|cod|trout|sardine|anchovy|seafood)\b"
DAIRY_EGGS = r"\b(egg|eggs|yogurt|yoghurt|cheese|milk|butter|cream|honey)\b"


def ingredient_conflict(text, preference):
    """Conservative text screen, not an ingredient/allergen certification."""
    text = text.lower()
    # Remove explicitly named plant alternatives before screening.
    text = re.sub(r"\b(?:vegan|plant-based|dairy-free|non-dairy)\s+(?:yogurt|yoghurt|cheese|milk|butter|cream)\b", "", text)
    text = re.sub(r"\b(?:almond|soy|soya|oat|coconut|peanut|cashew|sunflower)\s+(?:milk|butter|yogurt|cream)\b", "", text)
    patterns = []
    if preference != 'Omnivore':
        patterns.append(MEAT)
    if preference in ('Vegan', 'Vegetarian'):
        patterns.append(FISH)
    if preference == 'Vegan':
        patterns.append(DAIRY_EGGS)
    return any(re.search(pattern, text) for pattern in patterns)


DEFAULT_DATA = Path(__file__).with_name('Food_and_Nutrition__.csv')


def number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be a finite number between {low} and {high}.')


@dataclass(frozen=True)
class Profile:
    age: float
    weight: float
    activity: str
    preference: str
    conditions: tuple[str, ...] = ()
    calorie_target: float | None = None

    def validate(self):
        number(self.age, 'Age', 18, 100)
        number(self.weight, 'Weight (kg)', 30, 300)
        if self.activity not in ACTIVITY:
            raise ValueError('Choose a supported activity level.')
        if self.preference not in PREFERENCES:
            raise ValueError('Choose a supported dietary preference.')
        if not isinstance(self.conditions, (list, tuple)) or any(not isinstance(c, str) for c in self.conditions):
            raise ValueError('Conditions must be a list or tuple of labels.')
        if self.calorie_target is not None:
            number(self.calorie_target, 'Calorie target', 1, 10000)


class DietRecommender:
    def __init__(self, data):
        required = ['Ages', 'Weight', 'Activity Level', 'Dietary Preference', 'Disease', *MEALS, *NUTRIENTS]
        missing = set(required) - set(data.columns)
        if missing:
            raise ValueError(f'Missing dataset columns: {sorted(missing)}')
        self.data = data.copy().reset_index(drop=True)
        if self.data.empty or self.data[required].isna().any().any():
            raise ValueError('Dataset must be nonempty and contain no missing required values.')
        for col in ['Ages', 'Weight', *NUTRIENTS]:
            self.data[col] = pd.to_numeric(self.data[col], errors='raise')
            values = self.data[col].to_numpy(float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f'{col} must contain finite nonnegative values.')
        if not self.data['Activity Level'].isin(ACTIVITY).all():
            raise ValueError('Dataset contains unknown activity levels.')
        if not self.data['Dietary Preference'].isin(PREFERENCES).all():
            raise ValueError('Dataset contains unknown dietary preferences.')
        self.labels = self.data['Disease'].map(lambda s: set(x.strip() for x in s.split(',') if x.strip()))
        self.conditions = sorted(set().union(*self.labels))
        self.features = ['Ages', 'Weight', 'Activity Level', *self.conditions]
        raw = np.column_stack([self.data['Ages'], self.data['Weight'], self.data['Activity Level'].map(ACTIVITY),
                               *[self.labels.map(lambda labels: float(c in labels)) for c in self.conditions]])
        self.mean = raw.mean(axis=0)
        self.scale = raw.std(axis=0)
        self.scale[self.scale == 0] = 1
        self.normalized = (raw - self.mean) / self.scale

    @classmethod
    def from_csv(cls, path=DEFAULT_DATA):
        return cls(pd.read_csv(path))

    def recommend(self, profile, top_k=5, calorie_tolerance=None, nutrient_max=None, nutrient_min=None):
        profile.validate()
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError('top_k must be an integer between 1 and 100.')
        unknown = set(profile.conditions) - set(self.conditions)
        if unknown:
            raise ValueError(f'Unknown condition labels: {sorted(unknown)}')
        mask = self.data['Dietary Preference'].eq(profile.preference).to_numpy()
        descriptions = self.data[MEALS].agg(' '.join, axis=1)
        mask &= ~descriptions.map(lambda text: ingredient_conflict(text, profile.preference)).to_numpy()
        if calorie_tolerance is not None:
            number(calorie_tolerance, 'Calorie tolerance', 0, 1)
            if profile.calorie_target is None:
                raise ValueError('A calorie target is required when setting a tolerance.')
            mask &= (self.data['Calories'] - profile.calorie_target).abs().to_numpy() <= profile.calorie_target * calorie_tolerance
        for limits, op in [(nutrient_max or {}, 'max'), (nutrient_min or {}, 'min')]:
            for nutrient, value in limits.items():
                if nutrient not in NUTRIENTS:
                    raise ValueError(f'Unknown nutrient: {nutrient}')
                number(value, nutrient, 0, 1000000)
                mask &= (self.data[nutrient].le(value) if op == 'max' else self.data[nutrient].ge(value)).to_numpy()
        for nutrient in (nutrient_max or {}).keys() & (nutrient_min or {}).keys():
            if nutrient_min[nutrient] > nutrient_max[nutrient]:
                raise ValueError(f'Minimum exceeds maximum for {nutrient}.')
        v = (np.array([profile.age, profile.weight, ACTIVITY[profile.activity],
                       *[float(c in profile.conditions) for c in self.conditions]]) - self.mean) / self.scale
        denominator = np.linalg.norm(self.normalized, axis=1) * np.linalg.norm(v)
        similarity = np.divide(self.normalized @ v, denominator,
                               out=np.zeros(len(self.data)), where=denominator != 0)
        result = self.data.loc[mask].copy()
        result['Similarity'] = similarity[mask]
        # Similarity is a ranking score, not a probability or medical confidence.
        result['Score'] = (result['Similarity'] + 1) / 2
        if profile.calorie_target is not None:
            result['Calorie Difference'] = (result['Calories'] - profile.calorie_target).abs()
            result['Calorie Match'] = np.clip(1 - result['Calorie Difference'] / profile.calorie_target, 0, 1)
            result['Score'] = 0.7 * result['Score'] + 0.3 * result['Calorie Match']
        result['Matched Labels'] = self.labels.loc[result.index].map(lambda labels: ', '.join(sorted(labels & set(profile.conditions))))
        return result.sort_values('Score', ascending=False, kind='stable').drop_duplicates(MEALS).head(top_k)

    def save(self, path):
        """Portable JSON artifact includes data and fitted preprocessing state."""
        payload = {'version': 1, 'data': json.loads(self.data.to_json(orient='records')),
                   'features': self.features, 'mean': self.mean.tolist(), 'scale': self.scale.tolist()}
        Path(path).write_text(json.dumps(payload, allow_nan=False), encoding='utf-8')

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if payload.get('version') != 1:
            raise ValueError('Unsupported model version.')
        model = cls(pd.DataFrame(payload['data']))
        if payload['features'] != model.features or not np.allclose(payload['mean'], model.mean) or not np.allclose(payload['scale'], model.scale):
            raise ValueError('Saved preprocessing does not match the dataset.')
        return model
