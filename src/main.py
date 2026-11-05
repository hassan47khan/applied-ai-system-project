"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")

    # Starter example profile
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print()
    print("=" * 60)
    print(f"  TOP {len(recommendations)} RECOMMENDATIONS")
    print(f"  Preferences: {user_prefs}")
    print("=" * 60)

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        reasons = [r.strip() for r in explanation.split(";")] if explanation else []

        print(f"\n#{rank}  {song['title']} - {song['artist']}")
        print(f"     Score: {score:.2f}")
        print(f"     Reasons:")
        for reason in reasons:
            print(f"       - {reason}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
