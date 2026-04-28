import re
from joblib import Parallel, delayed

import pymorphy3

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMOJI_RE = re.compile(
    "[\U00010000-\U0010ffff"
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff]",
    flags=re.UNICODE,
)
_MD_RE = re.compile(r"[*_`#|>\[\]~]")
_SPACES_RE = re.compile(r"\s+")

STOPWORDS = {
    "и",
    "в",
    "во",
    "не",
    "что",
    "он",
    "на",
    "я",
    "с",
    "со",
    "как",
    "а",
    "то",
    "все",
    "она",
    "так",
    "его",
    "но",
    "да",
    "ты",
    "к",
    "у",
    "же",
    "вы",
    "за",
    "бы",
    "по",
    "только",
    "ее",
    "мне",
    "было",
    "вот",
    "от",
    "меня",
    "еще",
    "нет",
    "о",
    "из",
    "ему",
    "теперь",
    "когда",
    "даже",
    "ну",
    "вдруг",
    "ли",
    "если",
    "уже",
    "или",
    "ни",
    "быть",
    "был",
    "него",
    "до",
    "вас",
    "нибудь",
    "опять",
    "уж",
    "вам",
    "ведь",
    "там",
    "потом",
    "себя",
    "ничего",
    "ей",
    "может",
    "они",
    "тут",
    "где",
    "есть",
    "надо",
    "ней",
    "для",
    "мы",
    "тебя",
    "их",
    "чем",
    "была",
    "сам",
    "чтоб",
    "без",
    "будто",
    "человек",
    "чего",
    "раз",
    "тоже",
    "себе",
    "под",
    "будет",
    "ж",
    "тогда",
    "кто",
    "этот",
    "того",
    "потому",
    "этого",
    "какой",
    "совсем",
    "ним",
    "здесь",
    "этом",
    "один",
    "почти",
    "мой",
    "тем",
    "чтобы",
    "нее",
    "сейчас",
    "были",
    "куда",
    "зачем",
    "всех",
    "никогда",
    "можно",
    "при",
    "наконец",
    "два",
    "об",
    "другой",
    "хоть",
    "после",
    "над",
    "больше",
    "тот",
    "через",
    "эти",
    "нас",
    "про",
    "всего",
    "них",
    "какая",
    "много",
    "разве",
    "три",
    "эту",
    "моя",
    "впрочем",
    "хорошо",
    "свою",
    "этой",
    "перед",
    "иногда",
    "лучше",
    "чуть",
    "том",
    "нельзя",
    "такой",
    "им",
    "более",
    "всегда",
    "конечно",
    "всю",
    "между",
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _URL_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = _MD_RE.sub(" ", text)
    text = _SPACES_RE.sub(" ", text)
    return text.strip()


def lemmatize_text(text: str, morph: pymorphy3.MorphAnalyzer) -> str:
    try:
        import razdel
    except ImportError:
        raise ImportError("razdel is required for lemmatization")

    text = clean_text(text)
    tokens = [t.text for t in razdel.tokenize(text)]
    lemmas = []
    for token in tokens:
        token_lower = token.lower()
        if len(token_lower) < 3:
            continue
        if token_lower in STOPWORDS:
            continue
        if not token_lower.isalpha():
            continue
        lemma = morph.parse(token_lower)[0].normal_form
        lemmas.append(lemma)
    return " ".join(lemmas)


def _process_chunk(texts: list[str]) -> tuple[list[str], list[str]]:
    morph = pymorphy3.MorphAnalyzer()
    clean_results = [clean_text(t) for t in texts]
    lemm_results = [lemmatize_text(t, morph) for t in texts]
    return clean_results, lemm_results


def preprocess_texts(
    texts: list[str],
    n_jobs: int = 4,
    chunk_size: int = 10_000,
) -> tuple[list[str], list[str]]:
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_process_chunk)(chunk) for chunk in chunks
    )
    clean_texts: list[str] = []
    lemm_texts: list[str] = []
    for clean_chunk, lemm_chunk in results:
        clean_texts.extend(clean_chunk)
        lemm_texts.extend(lemm_chunk)
    return clean_texts, lemm_texts
