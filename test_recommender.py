import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from recommender import DietRecommender, Profile, PREFERENCES, MEALS, ingredient_conflict


class RecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DietRecommender.from_csv()

    def profile(self, **changes):
        fields = dict(age=42, weight=103, activity='Moderately Active', preference='Vegan', conditions=('Acne',))
        fields.update(changes)
        return Profile(**fields)

    def test_original_vegan_regression(self):
        result = self.model.recommend(self.profile())
        self.assertEqual(len(result), 5)
        self.assertTrue(result['Dietary Preference'].eq('Vegan').all())

    def test_preferences_across_profiles(self):
        for pref in PREFERENCES:
            for age, weight, activity in [(18, 40, 'Sedentary'), (65, 80, 'Lightly Active'), (80, 150, 'Extremely Active')]:
                result = self.model.recommend(self.profile(age=age, weight=weight, activity=activity, preference=pref))
                self.assertTrue(result['Dietary Preference'].eq(pref).all())
                self.assertTrue(np.isfinite(result['Score']).all())

    def test_calorie_and_nutrient_filters(self):
        result = self.model.recommend(self.profile(calorie_target=2000), calorie_tolerance=.1,
                                      nutrient_max={'Sugar': 150}, nutrient_min={'Protein': 60})
        self.assertFalse(result.empty)
        self.assertTrue(result['Calories'].between(1800, 2200).all())
        self.assertTrue(result['Sugar'].le(150).all())
        self.assertTrue(result['Protein'].ge(60).all())

    def test_no_results_does_not_relax_filters(self):
        self.assertTrue(self.model.recommend(self.profile(), nutrient_max={'Calories': 0}).empty)

    def test_mislabeled_animal_ingredients_are_excluded(self):
        data = self.model.data[self.model.data['Dietary Preference'].eq('Vegan')].head(2).copy()
        data.loc[data.index[0], 'Lunch Suggestion'] = 'Grilled chicken salad'
        for meal in MEALS:
            data.loc[data.index[1], meal] = 'Lentils with vegetables'
        result = DietRecommender(data).recommend(self.profile(conditions=()))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['Lunch Suggestion'], 'Lentils with vegetables')
        self.assertFalse(ingredient_conflict('Apple with almond butter and soy milk', 'Vegan'))
        self.assertTrue(ingredient_conflict('Greek yogurt with eggs', 'Vegan'))
        self.assertFalse(ingredient_conflict('Salmon with vegetables', 'Pescatarian'))
        self.assertTrue(ingredient_conflict('Chicken salad', 'Pescatarian'))

    def test_invalid_profiles(self):
        for changes in [{'age': -1}, {'weight': float('nan')}, {'weight': float('inf')},
                        {'activity': 'unknown'}, {'preference': 'unknown'}, {'conditions': ('Unknown',)},
                        {'calorie_target': 0}, {'age': True}, {'conditions': 'Acne'}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.model.recommend(self.profile(**changes))

    def test_invalid_constraints(self):
        for kwargs in [{'top_k': 0}, {'top_k': True}, {'calorie_tolerance': .1},
                       {'nutrient_max': {'Unknown': 1}}, {'nutrient_min': {'Protein': -1}},
                       {'nutrient_min': {'Protein': 80}, 'nutrient_max': {'Protein': 50}}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model.recommend(self.profile(), **kwargs)

    def test_unique_plans_and_stable_order(self):
        first = self.model.recommend(self.profile(), top_k=100)
        self.assertFalse(first.duplicated(MEALS).any())
        pd.testing.assert_frame_equal(first, self.model.recommend(self.profile(), top_k=100))

    def test_model_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.json'
            self.model.save(path)
            loaded = DietRecommender.load(path)
            pd.testing.assert_frame_equal(self.model.recommend(self.profile()), loaded.recommend(self.profile()), check_dtype=False)

    def test_invalid_dataset(self):
        for data in [self.model.data.iloc[:0], self.model.data.drop(columns='Calories'),
                     self.model.data.assign(Calories=float('nan'))]:
            with self.assertRaises(ValueError):
                DietRecommender(data)

    def test_zero_vector_scores_remain_finite(self):
        data = self.model.data.iloc[[0, 0]].copy()
        model = DietRecommender(data)
        row = data.iloc[0]
        result = model.recommend(Profile(float(row.Ages), float(row.Weight), row['Activity Level'],
                                        row['Dietary Preference'], tuple(model.conditions)))
        self.assertTrue(np.isfinite(result['Score']).all())


if __name__ == '__main__':
    unittest.main()
