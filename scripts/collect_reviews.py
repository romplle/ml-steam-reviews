import csv
import json
import time
from pathlib import Path
from urllib.parse import quote

import requests


APPREVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"
REVIEWS_PER_GAME = 1000
GAME_GROUP = "all"
LANGUAGE = "english"
REVIEW_FILTER = "all"
DAY_RANGE = 365
SLEEP_SECONDS = 0.5
TIMEOUT = 30
OUTPUT_PATH = Path("data/raw/steam_reviews_raw.csv")
RAW_JSON_DIR = Path("data/raw/json")

COMPETITIVE_GAMES = {
    "Counter-Strike 2": 730,
    "Dota 2": 570,
    "PUBG: Battlegrounds": 578080,
    "Apex Legends": 1172470,
    "Team Fortress 2": 440,
    "War Thunder": 236390,
}

OFFLINE_GAMES = {
    "The Witcher 3: Wild Hunt": 292030,
    "Red Dead Redemption 2": 1174180,
    "God of War": 1593500,
    "The Last of Us Part I": 1888930,
    "BioShock Infinite": 8870,
    "Black Myth: Wukong": 2358720,
}

GAME_GROUPS = {
    "competitive": COMPETITIVE_GAMES,
    "offline": OFFLINE_GAMES,
}

DEFAULT_GAMES = {
    **COMPETITIVE_GAMES,
    **OFFLINE_GAMES,
}

GAME_TO_GROUP = {
    **{game: "competitive" for game in COMPETITIVE_GAMES},
    **{game: "offline" for game in OFFLINE_GAMES},
}

CSV_COLUMNS = [
    "appid",
    "game",
    "game_group",
    "recommendationid",
    "review",
    "voted_up",
    "votes_up",
    "votes_funny",
    "weighted_vote_score",
    "comment_count",
    "steam_purchase",
    "received_for_free",
    "written_during_early_access",
    "timestamp_created",
    "author_num_games_owned",
    "author_num_reviews",
    "playtime_forever",
    "playtime_at_review",
]


def get_games_to_collect():
    if GAME_GROUP == "all":
        return DEFAULT_GAMES
    return GAME_GROUPS[GAME_GROUP]


def request_reviews_page(session, appid, cursor, language, review_filter, day_range, timeout):
    params = {
        "json": 1,
        "filter": review_filter,
        "language": language,
        "num_per_page": 100,
        "cursor": cursor,
        "review_type": "all",
        "purchase_type": "all",
    }
    if review_filter == "all":
        params["day_range"] = day_range

    response = session.get(
        APPREVIEWS_URL.format(appid=appid),
        params=params,
        timeout=timeout,
        headers={"User-Agent": "ml-steam-reviews-course-project/1.0"},
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("success") not in (1, True):
        raise RuntimeError(f"Steam returned unsuccessful response for appid={appid}: {payload}")
    return payload


def flatten_review(appid, game, game_group, item):
    author = item.get("author") or {}
    return {
        "appid": appid,
        "game": game,
        "game_group": game_group,
        "recommendationid": item.get("recommendationid"),
        "review": item.get("review"),
        "voted_up": item.get("voted_up"),
        "votes_up": item.get("votes_up"),
        "votes_funny": item.get("votes_funny"),
        "weighted_vote_score": item.get("weighted_vote_score"),
        "comment_count": item.get("comment_count"),
        "steam_purchase": item.get("steam_purchase"),
        "received_for_free": item.get("received_for_free"),
        "written_during_early_access": item.get("written_during_early_access"),
        "timestamp_created": item.get("timestamp_created"),
        "author_num_games_owned": author.get("num_games_owned"),
        "author_num_reviews": author.get("num_reviews"),
        "playtime_forever": author.get("playtime_forever"),
        "playtime_at_review": author.get("playtime_at_review"),
    }


def collect_for_game(
    session,
    appid,
    game,
    game_group,
    target_count,
    language,
    review_filter,
    day_range,
    sleep_seconds,
    timeout,
    raw_json_dir,
    seen_recommendation_ids,
):
    cursor = "*"
    rows = []
    page_number = 0
    raw_json_dir.mkdir(parents=True, exist_ok=True)
    safe_game = quote(game.replace(" ", "_"), safe="")
    raw_json_path = raw_json_dir / f"{appid}_{safe_game}.jsonl"

    with raw_json_path.open("w", encoding="utf-8") as raw_file:
        duplicate_only_pages = 0
        while len(rows) < target_count:
            page_number += 1
            payload = request_reviews_page(
                session=session,
                appid=appid,
                cursor=cursor,
                language=language,
                review_filter=review_filter,
                day_range=day_range,
                timeout=timeout,
            )
            reviews = payload.get("reviews") or []
            raw_file.write(
                json.dumps(
                    {
                        "appid": appid,
                        "game": game,
                        "game_group": game_group,
                        "page_number": page_number,
                        "cursor_in": cursor,
                        "cursor_out": payload.get("cursor"),
                        "filter": review_filter,
                        "day_range": day_range if review_filter == "all" else None,
                        "reviews": reviews,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if not reviews:
                print(f"[{game}] no more reviews after {len(rows)} collected")
                break

            new_count = 0
            for item in reviews:
                recommendation_id = str(item.get("recommendationid") or "")
                if not recommendation_id or recommendation_id in seen_recommendation_ids:
                    continue
                seen_recommendation_ids.add(recommendation_id)
                rows.append(flatten_review(appid=appid, game=game, game_group=game_group, item=item))
                new_count += 1
                if len(rows) >= target_count:
                    break

            print(f"[{game}] page={page_number} new={new_count} total={len(rows)}")
            duplicate_only_pages = duplicate_only_pages + 1 if new_count == 0 else 0
            if duplicate_only_pages >= 3:
                print(f"[{game}] stopped after 3 pages without new unique reviews")
                break
            next_cursor = payload.get("cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
            time.sleep(sleep_seconds)

    return rows


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    games = get_games_to_collect()
    session = requests.Session()
    all_rows = []
    seen_recommendation_ids = set()

    for game, appid in games.items():
        rows = collect_for_game(
            session=session,
            appid=appid,
            game=game,
            game_group=GAME_TO_GROUP[game],
            target_count=REVIEWS_PER_GAME,
            language=LANGUAGE,
            review_filter=REVIEW_FILTER,
            day_range=DAY_RANGE,
            sleep_seconds=SLEEP_SECONDS,
            timeout=TIMEOUT,
            raw_json_dir=RAW_JSON_DIR,
            seen_recommendation_ids=seen_recommendation_ids,
        )
        all_rows.extend(rows)

    write_csv(all_rows, OUTPUT_PATH)
    print(f"Saved {len(all_rows)} reviews to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
