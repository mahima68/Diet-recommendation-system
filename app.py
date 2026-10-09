"""Interactive terminal interface; also supports arguments and CSV export."""
import argparse
from recommender import DietRecommender, Profile, ACTIVITY, PREFERENCES


def main():
    parser = argparse.ArgumentParser(description='Find matching diet plans from the demo dataset.')
    parser.add_argument('--age', type=float)
    parser.add_argument('--weight', type=float)
    parser.add_argument('--activity', choices=ACTIVITY)
    parser.add_argument('--preference', choices=PREFERENCES)
    parser.add_argument('--conditions', default='', help='Comma-separated dataset labels')
    parser.add_argument('--calories', type=float)
    parser.add_argument('--tolerance', type=float, help='Relative calorie tolerance, e.g. 0.1 for 10%%')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--export', help='Output CSV path')
    args = parser.parse_args()
    try:
        age = args.age if args.age is not None else float(input('Age (18–100): '))
        weight = args.weight if args.weight is not None else float(input('Weight in kg: '))
        activity = args.activity or input(f'Activity ({", ".join(ACTIVITY)}): ').strip()
        preference = args.preference or input(f'Diet ({", ".join(PREFERENCES)}): ').strip()
        profile = Profile(age, weight, activity, preference,
                          tuple(c.strip() for c in args.conditions.split(',') if c.strip()), args.calories)
        result = DietRecommender.from_csv().recommend(profile, args.top_k, args.tolerance)
    except (ValueError, EOFError) as exc:
        parser.error(str(exc))
    print('\nEducational dataset demo. Nutrition units and dietary labels are unverified.')
    if result.empty:
        print('No plans satisfy your filters. Review the criteria; filters were not relaxed.')
    for rank, (_, row) in enumerate(result.iterrows(), 1):
        print(f'\nPlan {rank} | {row["Dietary Preference"]} | Dataset calories: {row["Calories"]}')
        for meal in ['Breakfast', 'Lunch', 'Dinner', 'Snack']:
            print(f'  {meal}: {row[meal + " Suggestion"]}')
        if args.calories is not None:
            print(f'  Difference from target: {row["Calorie Difference"]:g}')
        if row['Matched Labels']:
            print(f'  Matching dataset labels: {row["Matched Labels"]}')
    if args.export:
        result.to_csv(args.export, index=False)
        print(f'\nSaved results to {args.export}')


if __name__ == '__main__':
    main()
