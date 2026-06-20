from __future__ import annotations

from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


CATEGORIES = ("Sports", "Business", "Technology", "Politics")


TRAINING_DATA = [
    # Sports
    ("India wins the cricket series after a thrilling final over chase", "Sports"),
    ("Football club signs a new striker before the summer transfer deadline", "Sports"),
    ("Olympic champion breaks the world record in the 200 meter sprint", "Sports"),
    ("Tennis star advances to the grand slam semifinal after straight set win", "Sports"),
    ("Basketball team secures playoff berth with a dominant home victory", "Sports"),
    ("Hockey captain scores twice as the team lifts the championship trophy", "Sports"),
    ("Coach announces squad for upcoming cricket world cup tournament", "Sports"),
    ("Star bowler ruled out of test match due to a shoulder injury", "Sports"),
    ("Local marathon attracts thousands of runners across the city", "Sports"),
    ("Formula racing driver claims pole position in qualifying session", "Sports"),
    ("Badminton pair reaches final after defeating defending champions", "Sports"),
    ("Club manager praises defense after clean sheet in league match", "Sports"),
    ("Young athlete wins gold medal at national swimming championship", "Sports"),
    ("Fans celebrate dramatic penalty shootout win in football final", "Sports"),
    ("Cricket board announces fixtures for the domestic league season", "Sports"),
    ("Wrestler qualifies for international tournament after national trials", "Sports"),
    ("Baseball pitcher delivers complete game shutout against rivals", "Sports"),
    ("Volleyball team wins regional title with unbeaten season", "Sports"),
    ("Cyclist takes mountain stage lead in the grand tour", "Sports"),
    ("Rugby side names experienced captain for the championship opener", "Sports"),
    # Business
    ("Stock markets rise as investors expect strong quarterly earnings", "Business"),
    ("Central bank keeps interest rates unchanged amid inflation concerns", "Business"),
    ("Startup raises fresh funding to expand its online payment platform", "Business"),
    ("Automaker reports higher sales after demand improves in export markets", "Business"),
    ("Oil prices fall as global supply outlook improves", "Business"),
    ("Retail chain opens new stores after revenue growth beats forecasts", "Business"),
    ("Company shares drop following weak profit guidance for the next quarter", "Business"),
    ("Merger deal creates one of the largest telecom companies in the region", "Business"),
    ("Government announces tax incentives for small and medium businesses", "Business"),
    ("Bank launches digital lending service for entrepreneurs", "Business"),
    ("Airline posts record profit after travel demand rebounds", "Business"),
    ("Manufacturing output expands as factories receive more export orders", "Business"),
    ("Real estate prices climb despite tighter home loan rules", "Business"),
    ("Ecommerce platform cuts delivery fees during festive sale period", "Business"),
    ("Rupee weakens against dollar as foreign investors sell equities", "Business"),
    ("Food delivery company files papers for public market listing", "Business"),
    ("Luxury brand forecasts slower growth in key overseas markets", "Business"),
    ("Insurance firm acquires fintech startup to strengthen digital products", "Business"),
    ("Corporate bond yields ease after strong demand from institutions", "Business"),
    ("Trade deficit narrows as exports of electronics and textiles increase", "Business"),
    # Technology
    ("New smartphone features an AI camera and faster mobile processor", "Technology"),
    ("Cybersecurity researchers discover a critical vulnerability in cloud software", "Technology"),
    ("Tech company releases open source tools for machine learning developers", "Technology"),
    ("Electric vehicle maker updates battery software to improve driving range", "Technology"),
    ("Scientists build a quantum chip that improves error correction", "Technology"),
    ("Social media app launches encrypted messaging for all users", "Technology"),
    ("Satellite internet service expands coverage to remote villages", "Technology"),
    ("Robotics startup unveils warehouse automation system for logistics firms", "Technology"),
    ("Data center operators invest in cooling systems for AI workloads", "Technology"),
    ("New laptop lineup uses more efficient chips and brighter displays", "Technology"),
    ("Streaming platform tests recommendation algorithm for personalized playlists", "Technology"),
    ("Developers patch security bug affecting millions of web servers", "Technology"),
    ("Drone delivery trials begin using autonomous navigation technology", "Technology"),
    ("Cloud provider introduces cheaper storage tier for enterprise customers", "Technology"),
    ("Researchers train language model to detect medical imaging errors", "Technology"),
    ("Gaming console update adds support for high refresh rate monitors", "Technology"),
    ("Semiconductor company announces advanced processor manufacturing roadmap", "Technology"),
    ("Password manager adds passkey support for desktop and mobile apps", "Technology"),
    ("Augmented reality headset receives new hand tracking features", "Technology"),
    ("Search engine rolls out generative AI summaries for news queries", "Technology"),
    # Politics
    ("Parliament debates new bill on education reform and public spending", "Politics"),
    ("Prime minister meets opposition leaders before the budget session", "Politics"),
    ("Election commission announces dates for state assembly polls", "Politics"),
    ("Senate committee questions ministers over national security policy", "Politics"),
    ("Mayor unveils transport plan ahead of the city council vote", "Politics"),
    ("Government faces criticism over proposed changes to immigration rules", "Politics"),
    ("Political parties release manifestos before the general election", "Politics"),
    ("Cabinet approves new welfare scheme for rural households", "Politics"),
    ("Diplomats hold talks to ease tensions along the border", "Politics"),
    ("Lawmakers call for inquiry into public procurement contracts", "Politics"),
    ("President signs climate legislation after months of negotiation", "Politics"),
    ("Opposition demands debate on unemployment during parliamentary session", "Politics"),
    ("State governor appoints new ministers after coalition reshuffle", "Politics"),
    ("Court hears petition challenging campaign finance regulations", "Politics"),
    ("Foreign minister visits neighboring country for trade and security talks", "Politics"),
    ("Voters line up early as local elections begin across districts", "Politics"),
    ("Ruling party nominates candidate for upcoming presidential race", "Politics"),
    ("Policy panel recommends reforms to strengthen local government", "Politics"),
    ("Officials announce peace agreement after diplomatic negotiations", "Politics"),
    ("Legislature passes transparency law for public office holders", "Politics"),
]


@lru_cache(maxsize=1)
def train_model() -> Pipeline:
    texts, labels = zip(*TRAINING_DATA)
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(texts, labels)
    return model


def predict_category(text: str) -> tuple[str, float, dict[str, float]]:
    cleaned_text = " ".join(text.split())
    if not cleaned_text:
        raise ValueError("Please enter a headline or short article.")

    model = train_model()
    predicted_label = model.predict([cleaned_text])[0]
    probabilities = model.predict_proba([cleaned_text])[0]
    classes = model.classes_
    scores = {label: float(score) for label, score in zip(classes, probabilities)}
    confidence = scores[predicted_label]
    return predicted_label, confidence, scores


if __name__ == "__main__":
    samples = [
        "The cricket team won the final after a brilliant chase",
        "Shares rose after the company reported record profit",
        "A new AI chip improves cloud computing performance",
        "Parliament passed a major election reform bill",
    ]

    for sample in samples:
        label, confidence, _ = predict_category(sample)
        print(f"{label:10} {confidence:.2%} - {sample}")
