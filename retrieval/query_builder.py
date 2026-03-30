import re

from models.schema import Claim


_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would could should"
    " may might shall of in on at to for with by from that this these those it its we our"
    " this paper paper method approach results result model models".split()
)

_BENCHMARK_RE = re.compile(
    r"\b(BLEU|ROUGE|WMT|SQuAD|GLUE|SuperGLUE|ImageNet|COCO|MNIST|CIFAR|MMLU|"
    r"HumanEval|GSM8K|ARC|TruthfulQA|HellaSwag|F1|mAP|accuracy|perplexity)\b",
    re.IGNORECASE,
)


def extract_benchmark_terms(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _BENCHMARK_RE.findall(text):
        if match.lower() in seen:
            continue
        seen.add(match.lower())
        ordered.append(match)
    return ordered


def _extract_keywords(text: str, top_n: int = 6) -> str:
    words = re.findall(r"\b[A-Za-z][A-Za-z\-]{2,}\b", text)
    keywords = [word for word in words if word.lower() not in _STOPWORDS]
    unique: list[str] = []
    seen: set[str] = set()
    for word in keywords:
        lowered = word.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(word)
    return " ".join(unique[:top_n])


def _shorten_claim(text: str, max_chars: int = 180) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    shortened = sentences[0] if sentences else text
    return shortened[:max_chars]


def build_query_plan(claim: Claim, paper_title: str = "") -> dict[str, list[str]]:
    keywords = _extract_keywords(claim.text, top_n=6)
    benchmark_terms = extract_benchmark_terms(claim.text)
    base = _shorten_claim(claim.text)

    scholarly_queries: list[str] = []
    general_queries: list[str] = []

    def add(target: list[str], query: str) -> None:
        if not query:
            return
        if query not in target:
            target.append(query.strip())

    add(scholarly_queries, f"{base} research paper")
    if paper_title:
        add(scholarly_queries, f"{paper_title[:80]} {keywords} research")
    if benchmark_terms:
        add(scholarly_queries, f"{benchmark_terms[0]} {keywords} research")

    add(general_queries, base)
    if benchmark_terms:
        add(general_queries, f"{benchmark_terms[0]} {keywords}")
    if paper_title:
        add(general_queries, f"{paper_title[:80]} {keywords}")
    if claim.claim_type == "contribution":
        add(general_queries, f"{keywords} prior work research")

    return {
        "scholarly": scholarly_queries[:3],
        "general": general_queries[:3],
        "benchmark_terms": benchmark_terms,
    }
