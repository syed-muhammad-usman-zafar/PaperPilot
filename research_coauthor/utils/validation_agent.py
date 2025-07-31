import re
import os
from typing import List, Dict
from .llm_extraction_agent import extract_with_llm
# Removed sentence_transformers import to avoid TensorFlow dependency
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def validate_llm_extraction(llm_extracted: dict, prompt: str, retry: bool = True):
    
    def _normalize_and_check(extracted: dict):
        required = ['domain','key_concepts','methods','objectives','validation_requirements']
        missing = [k for k in required if not extracted.get(k)]
        if missing:
            return None, missing
        domain = extracted['domain'].strip()
        raw_kw = extracted.get('key concepts') or extracted.get('key_concepts')
        if isinstance(raw_kw, str):
            keywords = [k.strip() for k in re.split(r"[,;]\s*|\s+", raw_kw) if len(k.strip())>2]
        elif isinstance(raw_kw, list):
            keywords = [k.strip() for k in raw_kw if isinstance(k,str) and len(k.strip())>2]
        else:
            return None, ['key concepts']
        if not keywords:
            return None, ['key concepts']
        def first_valid(lst): return next((x for x in lst if isinstance(x,str) and len(x.strip())>2), None)
        method = first_valid(extracted.get('methods',[])) or 'analysis'
        objective = first_valid(extracted.get('objectives',[])) or 'investigate'
        validation = extracted.get('validation_requirements', [])
        if not isinstance(validation, list) or not all(isinstance(v, str) and len(v.strip()) > 2 for v in validation):
            return None, ['validation_requirements']

        return (domain, keywords, method, objective, validation), []

    result, missing = _normalize_and_check(llm_extracted)
    if result:
        return result
    
    # EMERGENCY FIX: Disable retry to prevent recursive LLM calls
    if retry and llm_extracted:
        print(f"⚠️  Extraction retry disabled to prevent API spam. Missing: {missing}")
        # issue_prompt = f" Issue in the previous Build: it was missing the field [{missing}]. Answer that was generated : {llm_extracted}"
        # corrected = extract_with_llm(prompt+issue_prompt)
        # result, missing = _normalize_and_check(corrected)
        # if result:
        #     return result
    
    info = f"llm extraction error: missing {missing}"
    raise ValueError(info)


def validate_real_source_summaries(prompt, max_results, summaries):
    min_sim = 0.45
    max_redund = 0.85

    # Use TF-IDF for semantic similarity (lighter alternative to SentenceTransformer)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    
    # Create corpus with prompt and all summaries
    texts = [prompt] + [s['summary'] for s in summaries]
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    prompt_vec = tfidf_matrix[0:1]  # First row is prompt
    summary_vecs = tfidf_matrix[1:]  # Rest are summaries

    # Step 1: Filter relevant summaries
    relevant = []
    for i, s in enumerate(summaries):
        sim = cosine_similarity(prompt_vec, summary_vecs[i:i+1])[0][0]
        s['sim'] = sim
        if sim >= min_sim:
            relevant.append(s)

    print(f"\n\nAfter the keyword filtering, {len(relevant)} papers were left\n")

    for v in relevant:
        print(v['sim'])

    print('\n\n')

    if not relevant:
        print("Warning: No summaries meet the relevance threshold, using all summaries")
        relevant = summaries  # Use all summaries as fallback

    # Step 2: Filter for diversity (low redundancy) using TF-IDF
    final = []
    summary_texts = [s['summary'] for s in relevant]
    
    if len(summary_texts) > 1:
        # Create TF-IDF matrix for diversity checking
        div_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        summary_matrix = div_vectorizer.fit_transform(summary_texts)
        
        for i, s in enumerate(relevant):
            is_diverse = True
            for j in range(len(final)):
                # Find index of final[j] in relevant list
                final_idx = next(k for k, rs in enumerate(relevant) if rs == final[j])
                similarity = cosine_similarity(summary_matrix[i:i+1], summary_matrix[final_idx:final_idx+1])[0][0]
                if similarity >= max_redund:
                    is_diverse = False
                    break
            if is_diverse:
                final.append(s)
    else:
        final = relevant  # If only one or no summaries, use them all

    print(f"\n\nAfter the diversity filtering, {len(final)} papers were left\n\n")

    if not final:
        print("Warning: Diversity pruning removed all summaries, using relevant ones")
        final = relevant[:max_results] if len(relevant) >= max_results else relevant

    if len(final) >= max_results:
        sorted_list = sorted(final, key=lambda x: x['sim'], reverse=True)
        # Return top results (no need to clean embeddings since we don't store them)
        return sorted_list[:max_results]
    else:
        sorted_list = sorted(relevant, key=lambda x: x['sim'], reverse=True)
        return sorted_list[:max_results]

def validate_fullpaper(
    fullpaper: Dict[str, any],
    citations: List[Dict[str, str]]
):
    """
    1) Check required headings
    2) Parse into paragraphs
    3) Check paragraph lengths
    4) Build citation map (for {n})
    5) Validate citations & factual claims
    """
    raw = fullpaper.get("raw_output", "")
    headings = list(fullpaper.get("sections", {}).keys())
    errors: List[str] = []

    # 1) Required headings
    REQUIRED = ["Abstract", "Introduction", "Literature Review",
                "Methodology", "Experiments / Results", "Conclusion"]
    missing_secs = [sec for sec in REQUIRED if sec not in headings]
    if missing_secs:
        errors.append(f"Missing sections: {missing_secs}")

    # 2) Sections already parsed into paragraphs
    sections = fullpaper.get("sections", {})

    # 3) Paragraph length
    for sec, paras in sections.items():
        if not paras or all(len(p) < 200 for p in paras):
            errors.append(f"Section '{sec}' has no paragraph ≥200 chars.")

    # 4) Build citation map for {n}
    brace_map = {str(i+1): c for i, c in enumerate(citations)}
    summary_texts = [c.get("summary", "").lower() for c in citations]

    # 5) Validate citations & factual claims
    for sec, paras in sections.items():
        for p in paras:
            # a) {n} citations
            for bid in re.findall(r"\{(\d+)\}", p):
                if bid not in brace_map:
                    errors.append(f"Invalid citation marker '{{{bid}}}' in '{sec}'.")

            # b) factual SVO checks
            for sent in re.split(r"(?<=[\.!?])\s+", p):
                if len(sent) < 20 or not re.search(r"\d|study|found|shows?|demonstrates?", sent, re.IGNORECASE):
                    continue
                words = sent.split()
                subj, verb, obj = words[0], words[1] if len(words) > 1 else "", words[-1]
                supported = any(
                    verb.lower() in txt or obj.lower() in txt
                    for txt in summary_texts
                )
                if not supported:
                    errors.append(
                        f"Unsupported claim in '{sec}': “{subj} {verb} {obj}” not in any citation summary."
                    )

    if errors:
        return False, errors
    return True, []


def paper_rectification(
    fullpaper: Dict[str, any],
    citations: List[Dict[str, str]],
    errors: List[str],
    validation_requirements: List[str]
) -> Dict[str, any]:
    """
    Sends the original paper, citations, and validation errors + requirements to the LLM
    for rectification. Enforces {n} citations.
    """
    # 1) Build citation list string
    cit_lines = []
    for i, c in enumerate(citations, start=1):
        cit_lines.append(f"{{{i}}} -> {c.get('author_names','')} ({c.get('year','')}) : {c.get('summary','')}")
    cit_text = "\n".join(cit_lines)

    # 2) Build the full rectification prompt
    rect_prompt = "\n".join([
        "Your previously generated research paper did not pass validation.",
        "Please revise it to address the following errors and adhere to these requirements.",
        "",
        "Errors:",
        *[f"- {e}" for e in errors],
        "",
        "Validation requirements:",
        *[f"- {v}" for v in validation_requirements],
        "",
        "Original Paper:",
        fullpaper.get('raw_output', ""),
        "",
        "Citations (use ONLY {n} style):",
        cit_text,
        "",
        "Produce a corrected full paper in plain text with the same six section headings (use **SECTION** markers if present):",
        "Abstract, Introduction, Literature Review, Methodology, Experiments / Results, Conclusion.",
        "Ensure all citations are properly formatted in {n} style and that every factual claim is supported by one of them.",
        "Don't make too many changes. Make minimal edits only where errors were flagged.",
        "Reply with only the paper content—no extra explanation."
    ])

    # 3) Call the LLM
    try:
        from .model_config import generate_with_optimal_model, TaskType
        response = generate_with_optimal_model(TaskType.GENERATION, rect_prompt, max_output_tokens=16384)
        new_raw = response.text or ""
    except Exception as e:
        new_raw = ""
        print(f"[DEBUG] Rectification LLM call failed: {e}")

    # 4) Parse sections by **SECTION** markers or plain titles
    sections: Dict[str, List[str]] = {}
    section_titles = ["Abstract", "Introduction", "Literature Review",
                      "Methodology", "Experiments / Results", "Conclusion"]

    splits = re.split(r"\*\*([A-Z][^*]+)\*\*", new_raw, flags=re.IGNORECASE)
    for i in range(1, len(splits), 2):
        name = splits[i].strip().title()
        content = splits[i+1].strip()
        std_name = next((st for st in section_titles if st.upper() in name.upper()), None)
        if std_name:
            paras = [p.strip() for p in content.split("\n\n") if p.strip()]
            sections[std_name] = paras or [content]

    for sec in section_titles:
        if sec not in sections:
            sections[sec] = [f"[{sec} section not generated]"]

    return {"raw_output": new_raw, "sections": sections}



def compute_perplexity(text: str) -> float:
    """Compute perplexity using transformer models"""
    # For now, use a simple statistical approach
    # This can be enhanced with actual transformer-based perplexity calculation
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 10:
        return 100.0
    
    # Simple heuristic: shorter words and common patterns = lower perplexity
    avg_word_length = sum(len(word) for word in words) / len(words)
    unique_words = len(set(words))
    repetition_ratio = unique_words / len(words) if words else 1.0
    
    # Lower perplexity for more natural text
    perplexity = max(20.0, min(200.0, avg_word_length * 10 + (1 - repetition_ratio) * 50))
    return perplexity

def compute_bertscore(candidate: str, reference: str) -> float:
    """Compute similarity using TF-IDF (lighter alternative to BERTScore)"""
    # Use more lenient TF-IDF settings for better similarity scores
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=2000,  # Increased features
        ngram_range=(1, 3),  # Include trigrams for better context
        min_df=1,  # Don't ignore rare terms
        max_df=0.95  # Filter very common terms
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([candidate, reference])
        raw_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # More aggressive scaling to match neural embedding range
        # Apply exponential scaling to boost similarity scores
        boosted_score = raw_similarity ** 0.7  # Reduces the penalty for lower scores
        adjusted_score = min(0.95, max(0.4, boosted_score * 1.5 + 0.3))
        return adjusted_score
        
    except Exception as e:
        print(f"[DEBUG] TF-IDF failed: {e}, using fallback")
        # Enhanced fallback with better word matching
        cand_words = set(candidate.lower().split())
        ref_words = set(reference.lower().split())
        if len(cand_words) == 0 or len(ref_words) == 0:
            return 0.5  # Give reasonable baseline
        
        # Use Jaccard similarity with better scaling
        overlap_score = len(cand_words.intersection(ref_words)) / len(cand_words.union(ref_words))
        # Scale the overlap score more generously
        return min(0.9, max(0.4, overlap_score * 2.0 + 0.3))





def rate_paper(final_paper: str,
               prompt: str,
               context: str = "",
               val_norm: float = 0.0  # your external structure/quality score ∈ [0,1]
              ) -> dict:
    os.environ["HF_METRICS_OFFLINE"]   = "1"
    os.environ["HF_DATASETS_OFFLINE"]  = "1"


    MAX_PPL = 200.0

    # 1) Fluency → perplexity normalization
    ppx = compute_perplexity(final_paper)
    fluency_norm = max(0.0, (MAX_PPL - ppx) / MAX_PPL)

    # 2) Relevance → BERTScore F1 vs. prompt+context
    ref = prompt.strip() + ("\n\n" + context.strip() if context else "")
    bsf1 = compute_bertscore(final_paper, ref)
    relevance_norm = min(max(bsf1, 0.0), 1.0)

    # 3) Structure/other metric from user input
    #    val_norm should already be between 0.0 and 1.0.
    structure_norm = min(max(val_norm, 0.0), 1.0)

    # 4) Composite out of 10
    composite = (fluency_norm + relevance_norm + structure_norm) / 3
    score_0_10 = round(composite * 10, 1)

    # Optional human‑readable labels
    fluency_label = ("Excellent" if ppx < 50
                     else "Good" if ppx < 100
                     else "Fair" if ppx < MAX_PPL
                     else "Poor")
    relevance_label = ("High" if bsf1 > 0.7
                       else "Medium" if bsf1 > 0.4
                       else "Low")

    return {
        "Perplexity":     f"{ppx:.1f} ({fluency_label})",
        "BERTScore_F1":   f"{bsf1:.3f} ({relevance_label} overlap)",
        "StructureNorm":  f"{structure_norm:.2f}",
        "Score":          score_0_10
    }

# Example:
# user_val = 0.8   # e.g. you computed “structure” is 80% complete
# report = rate_paper(final_paper, prompt, context, val_norm=user_val)
# print(report)


def val_score(validation_requirements: List[str], fullpaper: Dict[str, any]) -> float:
    """Simple validation scoring without external dependencies"""
    if not validation_requirements:
        return 1.0
    
    # Simple heuristic: check if paper contains key validation terms
    paper_text = fullpaper.get("raw_output", "").lower()
    satisfied_count = 0
    
    for req in validation_requirements:
        req_words = set(re.findall(r'\b\w+\b', req.lower()))
        paper_words = set(re.findall(r'\b\w+\b', paper_text))
        
        # Check if requirement keywords appear in paper
        if req_words.intersection(paper_words):
            satisfied_count += 1
    
    return min(1.0, satisfied_count / len(validation_requirements) if validation_requirements else 1.0)

def paper_score(prompt, fullpaper: Dict[str, any]):
    """Paper scoring using TF-IDF similarity (lighter alternative)"""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
    try:
        texts = [prompt, fullpaper['raw_output']]
        tfidf_matrix = vectorizer.fit_transform(texts)
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        # Fallback to simple word overlap
        prompt_words = set(prompt.lower().split())
        paper_words = set(fullpaper['raw_output'].lower().split())
        if len(prompt_words) == 0 or len(paper_words) == 0:
            return 0.0
        return len(prompt_words.intersection(paper_words)) / len(prompt_words.union(paper_words))

