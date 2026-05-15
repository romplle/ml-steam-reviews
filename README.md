# Steam Review Experience Classifier

Задача проекта — по тексту Steam-отзыва и простым публичным метаданным предсказать примерный игровой опыт автора. Данные собираются собственным скриптом из Steam, затем классическими методами ML предсказывается примерный игровой опыт автора по тексту отзыва и публичным метаданным.


## Быстрый старт

```powershell
python -m pip install -r requirements.txt
python scripts/collect_reviews.py # Сбор данных
python scripts/prepare_dataset.py # Подготовка данных
jupyter notebook
```

Ноутбуки:

- `notebooks/01_eda.ipynb` - разведочный анализ данных.
- `notebooks/02_modeling.ipynb` - обучение моделей, метрики и анализ ошибок.

## Цель и метрики

Цель: предсказать класс игрового опыта пользователя по Steam-отзыву.

Целевая переменная: `experience_class`.

Основные метрики:

- `macro F1` - главная метрика, потому что классы несбалансированы.
- `accuracy` - дополнительная метрика.

## Данные

Данные собираются через публичный Steam endpoint:

`https://store.steampowered.com/appreviews/<appid>?json=1`

Текущие настройки сбора:

- `REVIEWS_PER_GAME = 1000`
- `GAME_GROUP = "all"`
- `LANGUAGE = "english"`
- `REVIEW_FILTER = "all"`
- `DAY_RANGE = 365`
- `review_type = all`
- `purchase_type = all`

Сохраняются только публичные поля:

- `appid`, `game`, `game_group`
- `recommendationid`, `review`
- `voted_up`, `votes_up`, `votes_funny`, `weighted_vote_score`, `comment_count`
- `steam_purchase`, `received_for_free`, `written_during_early_access`
- `timestamp_created`
- `author_num_games_owned`, `author_num_reviews`
- `playtime_forever`, `playtime_at_review`

## Группы игр

Игры разделены на две группы, потому что одинаковое количество часов означает разный опыт в одиночных и соревновательных играх.

`offline`:

- `The Witcher 3: Wild Hunt`
- `Red Dead Redemption 2`
- `God of War`
- `The Last of Us Part I`
- `BioShock Infinite`
- `Black Myth: Wukong`

`competitive`:

- `Counter-Strike 2`
- `Dota 2`
- `PUBG: Battlegrounds`
- `Apex Legends`
- `Team Fortress 2`
- `War Thunder`

## Классы опыта

Для `offline`:

| Класс | Диапазон часов |
|---|---:|
| `newbie` | до 10 |
| `casual` | 10-50 |
| `engaged` | 50-100 |
| `veteran` | 100+ |

Для `competitive`:

| Класс | Диапазон часов |
|---|---:|
| `newbie` | до 100 |
| `casual` | 100-500 |
| `engaged` | 500-2000 |
| `veteran` | 2000+ |

Класс строится по `playtime_at_review`, если поле доступно, иначе по `playtime_forever`.

## Подготовка данных

Подготовка выполняется в `scripts/prepare_dataset.py`.

Шаги:

- удаляются дубликаты по `recommendationid`;
- чистятся переносы строк и лишние пробелы в тексте;
- удаляются пустые отзывы;
- числовые поля приводятся к числам;
- часы переводятся из минут в часы;
- создаётся `experience_class`;
- добавляются `review_length_chars` и `review_length_words` для EDA;
- удаляются слишком короткие отзывы.

Текущий фильтр коротких отзывов:

- `MIN_REVIEW_WORDS = 20`
- `MIN_REVIEW_CHARS = 100`

## Подход к моделированию

Используемые модели:

- `DummyClassifier(strategy="most_frequent")` - baseline.
- `TF-IDF + LogisticRegression` - text-only модель.
- `TF-IDF + LogisticRegression + metadata` - основная модель.

`LinearSVC` был исключён из финального ноутбука, потому что в экспериментах стабильно уступал логистической регрессии.

Текущие параметры основной модели после подбора:

```python
TfidfVectorizer(
    min_df=3,
    max_df=0.85,
    ngram_range=(1, 2),
    max_features=50000,
)

LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
)
```

Валидация:

- `train_test_split`
- `test_size=0.2`
- `stratify=y`
- `random_state=42`

Подбор параметров выполнялся через `GridSearchCV` с `f1_macro`.

## Признаки

Текстовые признаки:

- TF-IDF по словам и биграммам из `review`.

Метаданные в основной модели:

- `game`
- `game_group`
- `voted_up`
- `votes_up`
- `votes_funny`
- `weighted_vote_score`
- `comment_count`
- `steam_purchase`
- `received_for_free`
- `written_during_early_access`
- `author_num_games_owned`
- `author_num_reviews`

Из финальной модели убраны:

- `review_length_chars`
- `review_length_words`

Причина: при сравнении вариантов лучшая модель на общем срезе показала результат без признаков длины отзыва.

## Анализ ошибок

В `notebooks/02_modeling.ipynb` сохраняются:

- сводная таблица метрик по `all`, `offline`, `competitive`;
- сравнение `macro_f1` моделей по группам;
- confusion matrix для лучшей модели в каждом срезе;
- classification report для каждого среза;
- примеры неверных предсказаний;
- ключевые слова классов в text-only модели.

Ключевые слова смотрятся именно на text-only модели, потому что там коэффициенты логистической регрессии интерпретируются как вклад слов и n-грамм.

## Риски и ограничения

- Steam-отзывы часто мемные и эмоциональные, поэтому по ним сложно понять игровой опыт.
- `playtime_forever` может быть больше, чем опыт на момент написания отзыва, если `playtime_at_review` недоступен.
- Классы могут быть несбалансированы, поэтому `macro F1` важнее `accuracy`.

## Структура проекта

```text
ml-steam-reviews/
├── data/
│   ├── raw/                 # сырые данные, не коммитятся
│   └── processed/           # подготовленный датасет, не коммитится
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── scripts/
│   ├── collect_reviews.py
│   └── prepare_dataset.py
├── requirements.txt
└── README.md
```
