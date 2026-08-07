"""
caption_metrics.py
-------------------
NLP metrics used to evaluate the quality of captions retrieved by the
CLIP-descriptor module (manuscript Section: Performance analysis, Eqs. 18-21):
BLEU, METEOR, ROUGE-L, and CIDEr.

Where available, this module wraps the `pycocoevalcap` implementations
(the standard reference implementations used in the image-captioning
literature); otherwise it falls back to lightweight NLTK-based
approximations so the pipeline remains runnable without the full COCO
caption-eval toolkit.
"""

import math
from collections import Counter


def _ngrams(tokens, n):
    return list(zip(*[tokens[i:] for i in range(n)]))


def bleu_score(candidate: str, references, max_n: int = 4):
    """
    BLEU-N score (Eq. 18): BLEU = BP * exp(sum_n w_n * log(p_n))
    where p_n is the modified n-gram precision and BP is the brevity
    penalty. Falls back to a self-contained implementation if
    `nltk` is unavailable.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        cand_tokens = candidate.lower().split()
        ref_tokens = [r.lower().split() for r in references]
        weights = tuple([1.0 / max_n] * max_n)
        return sentence_bleu(ref_tokens, cand_tokens, weights=weights,
                              smoothing_function=SmoothingFunction().method1)
    except ImportError:
        return _bleu_fallback(candidate, references, max_n)


def _bleu_fallback(candidate, references, max_n=4):
    cand_tokens = candidate.lower().split()
    ref_token_lists = [r.lower().split() for r in references]

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = Counter(_ngrams(cand_tokens, n))
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        max_ref_counts = Counter()
        for ref_tokens in ref_token_lists:
            ref_ngrams = Counter(_ngrams(ref_tokens, n))
            for ng, cnt in ref_ngrams.items():
                max_ref_counts[ng] = max(max_ref_counts[ng], cnt)
        clipped = sum(min(cnt, max_ref_counts[ng]) for ng, cnt in cand_ngrams.items())
        total = sum(cand_ngrams.values())
        precisions.append(clipped / total if total > 0 else 0.0)

    if min(precisions) == 0:
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)

    cand_len = len(cand_tokens)
    closest_ref_len = min((len(r) for r in ref_token_lists),
                           key=lambda rl: (abs(rl - cand_len), rl))
    bp = 1.0 if cand_len > closest_ref_len else math.exp(1 - closest_ref_len / (cand_len + 1e-8))

    return bp * geo_mean


def rouge_l(candidate: str, reference: str) -> float:
    """
    ROUGE-L (Eq. 20): F-measure based on the Longest Common Subsequence
    (LCS) between candidate and reference token sequences.
    """
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()
    m, n = len(cand_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if cand_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]

    precision = lcs / m
    recall = lcs / n
    beta = 1.2
    if precision + recall == 0:
        return 0.0
    return ((1 + beta ** 2) * precision * recall) / (recall + beta ** 2 * precision + 1e-8)


def meteor_score(candidate: str, references):
    """
    METEOR (Eq. 19): harmonic-mean-based alignment score combining unigram
    precision and recall with a fragmentation penalty. Uses NLTK's
    reference implementation when available.
    """
    try:
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
        cand_tokens = candidate.lower().split()
        ref_token_lists = [r.lower().split() for r in references]
        return nltk_meteor(ref_token_lists, cand_tokens)
    except ImportError:
        # Lightweight unigram-F approximation when NLTK/wordnet data unavailable.
        cand_tokens = set(candidate.lower().split())
        best = 0.0
        for ref in references:
            ref_tokens = set(ref.lower().split())
            if not cand_tokens or not ref_tokens:
                continue
            overlap = len(cand_tokens & ref_tokens)
            precision = overlap / len(cand_tokens)
            recall = overlap / len(ref_tokens)
            if precision + recall == 0:
                continue
            f_mean = (10 * precision * recall) / (recall + 9 * precision + 1e-8)
            best = max(best, f_mean)
        return best


def _tf_idf_vector(tokens, doc_freq, num_docs):
    tf = Counter(tokens)
    vec = {}
    for ng, count in tf.items():
        idf = math.log(max(num_docs / (1 + doc_freq.get(ng, 0)), 1e-8))
        vec[ng] = count * idf
    return vec


def _cosine(vec_a, vec_b):
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cider_score(candidates, references_list, n=4):
    """
    CIDEr (Eq. 21): consensus-based captioning metric using TF-IDF weighted
    n-gram cosine similarity between candidate and reference captions,
    averaged over n-gram orders 1..n.

    Args:
        candidates: list of candidate caption strings (one per image).
        references_list: list of lists of reference captions per image.
    """
    doc_freq = [Counter() for _ in range(n)]
    num_docs = len(references_list)

    for refs in references_list:
        seen_ngrams = [set() for _ in range(n)]
        for ref in refs:
            tokens = ref.lower().split()
            for k in range(1, n + 1):
                for ng in _ngrams(tokens, k):
                    seen_ngrams[k - 1].add(ng)
        for k in range(n):
            for ng in seen_ngrams[k]:
                doc_freq[k][ng] += 1

    scores = []
    for cand, refs in zip(candidates, references_list):
        cand_tokens = cand.lower().split()
        per_n_scores = []
        for k in range(1, n + 1):
            cand_vec = _tf_idf_vector(_ngrams(cand_tokens, k), doc_freq[k - 1], num_docs)
            ref_sims = []
            for ref in refs:
                ref_tokens = ref.lower().split()
                ref_vec = _tf_idf_vector(_ngrams(ref_tokens, k), doc_freq[k - 1], num_docs)
                ref_sims.append(_cosine(cand_vec, ref_vec))
            per_n_scores.append(sum(ref_sims) / len(ref_sims) if ref_sims else 0.0)
        scores.append(10 * sum(per_n_scores) / n)

    return float(sum(scores) / len(scores)) if scores else 0.0


def evaluate_captions(candidates, references_list):
    """
    Compute all four caption metrics (BLEU-4, METEOR, ROUGE-L, CIDEr) for a
    batch of candidate/reference caption pairs, mirroring Table 5.
    """
    bleu4 = sum(bleu_score(c, r, max_n=4) for c, r in zip(candidates, references_list)) / len(candidates)
    meteor = sum(meteor_score(c, r) for c, r in zip(candidates, references_list)) / len(candidates)
    rouge = sum(max(rouge_l(c, ref) for ref in r) for c, r in zip(candidates, references_list)) / len(candidates)
    cider = cider_score(candidates, references_list)

    return {"BLEU-4": bleu4, "METEOR": meteor, "ROUGE-L": rouge, "CIDEr": cider}
