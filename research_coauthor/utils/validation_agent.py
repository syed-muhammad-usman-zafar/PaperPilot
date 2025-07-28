import re
from .llm_extraction_agent import extract_with_llm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
from evaluate import load
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import os


def validate_llm_extraction(llm_extracted: dict, prompt: str, retry: bool = True):
    """
    Validates and normalizes LLM extraction. Attempts one auto-correction retry if fields missing.
    Returns tuple (domain, keywords, method, objective, data_types, method_type, objective_scope).
    Raises ValueError with 'llm extraction error: info' on final failure.
    """
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
    if retry and llm_extracted:
        issue_prompt = f" Issue in the previous Build: it was missing the field [{missing}]. Answer that was generated : {llm_extracted}"
        corrected = extract_with_llm(prompt+issue_prompt)
        result, missing = _normalize_and_check(corrected)
        if result:
            return result
    info = f"llm extraction error: missing {missing}"
    raise ValueError(info)


def validate_real_source_summaries(prompt, max_results, summaries):
    min_sim = 0.45
    max_redund = 0.85

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2') 
    prompt_emb = model.encode([prompt])

    # Precompute all summary embeddings
    for s in summaries:
        s['embedding'] = model.encode([s['summary']])

    # Step 1: Filter relevant summaries
    relevant = []
    for s in summaries:
        sim = cosine_similarity(prompt_emb, s['embedding'])[0][0]
        s['sim']=sim
        if sim >= min_sim:
            relevant.append(s)

    print(f"\n\nAfter the keyword filtering, {len(relevant)} papers were left\n")

    for v in relevant:
        print(v['sim'])

    print('\n\n')

    if not relevant:
        raise ValueError("No summaries meet the relevance threshold.")

    # Step 2: Filter for diversity (low redundancy)
    final = []
    for s in relevant:
        if all(
            cosine_similarity(s['embedding'], o['embedding'])[0][0] < max_redund
            for o in final
        ):
            final.append(s)

    print(f"\n\nAfter the diversity filtering, {len(final)} papers were left\n\n")

    if not final:
        raise ValueError("Diversity pruning removed all summaries.")

    if len(final)>=max_results:
        sorted_list = sorted(final, key=lambda x: x['sim'], reverse=True)
        cleaned_output = []
        for d in sorted_list[:max_results]:
            cleaned = {k: v for k, v in d.items() if k != 'embedding'}
            cleaned_output.append(cleaned)
        return cleaned_output
    else:
        sorted_list = sorted(relevant, key=lambda x: x['sim'], reverse=True)
        cleaned_output = []
        for d in sorted_list[:max_results]:
            cleaned = {k: v for k, v in d.items() if k != 'embedding'}
            cleaned_output.append(cleaned)
        print("Not enough papers to meet the max results. Returning all papers.")
        return cleaned_output

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
    model = GPT2LMHeadModel.from_pretrained("gpt2",
    local_files_only=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2",
    local_files_only=True)
    model.eval()

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model(**enc, labels=enc["input_ids"])
    # loss is cross‑entropy; perplexity = exp(loss)
    return torch.exp(outputs.loss).item()

# example
# ppx = compute_perplexity(final_paper)
# print(f"Perplexity: {ppx:.1f}")





def compute_bertscore(pred: str, ref: str):
    bertscore = load("bertscore",
    local_files_only=True)
    results = bertscore.compute(predictions=[pred], references=[ref], lang="en")
    # F1 is the typical aggregate
    return results["f1"][0]

# reference = prompt + "\n\n" + context
# ref = prompt.strip() + "\n\n" + context.strip()
# f1 = compute_bertscore(final_paper, ref)
# print(f"BERTScore F1 vs. prompt+context: {f1:.3f}")

# assume compute_perplexity() and compute_bertscore() are defined above

def rate_paper(final_paper: str,
               prompt: str,
               context: str = "",
               val_norm: float = 0.0  # your external structure/quality score ∈ [0,1]
              ) -> dict:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
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
    prompt = "rate the following paper from 0 to 1 based on the validation requirements. just reply with the number. (if validarion requirements are not present, reply with 1). u must reply with a number.\n\n  Validation Requirements : ".join(validation_requirements) + "\n\n  Full Paper" + fullpaper.get("raw_output", "")
    try:
        # from .model_config import generate_with_optimal_model, TaskType
        # response = generate_with_optimal_model(TaskType.GENERATION, prompt, max_output_tokens=100)
        # new_raw = int(response.text) or 1
        from dotenv import load_dotenv
        import google.generativeai as genai
        from enum import Enum

        load_dotenv()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        response = genai.generate_content(prompt, model="gemini-1.5-flash", max_output_tokens=12)
        reply = float(response.text.strip()) or 0.5 
    except Exception as e:
        new_raw = 0.5
        print(f"[DEBUG] val score failed: {e}")
    return new_raw

def paper_score(prompt, fullpaper: Dict[str, any]):

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2') 
    prompt_emb = model.encode([prompt])
    content_emb = model.encode([fullpaper['raw_output']])

    return cosine_similarity(prompt_emb, content_emb)[0][0]

