import re
from .llm_extraction_agent import extract_with_llm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
from .writing_agent import model

def validate_llm_extraction(llm_extracted: dict, prompt: str, retry: bool = True):
    """
    Validates and normalizes LLM extraction. Attempts one auto-correction retry if fields missing.
    Returns tuple (domain, keywords, method, objective, data_types, method_type, objective_scope).
    Raises ValueError with 'llm extraction error: info' on final failure.
    """
    def _normalize_and_check(extracted: dict):
        required = ['domain','research methods','objectives','data types','key concepts','method_type','objective_scope', 'validation_requirements']
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
        method = first_valid(extracted.get('research methods',[])) or 'analysis'
        objective = first_valid(extracted.get('objectives',[])) or 'investigate'
        data_types = extracted.get('data types',[])
        if not isinstance(data_types,list):
            return None, ['data types']
        method_type = extracted.get('method_type',method).lower().strip()
        objective_scope = extracted.get('objective_scope',objective).lower().strip()
        allowed_m = {'empirical','theoretical','review','exploratory'}
        allowed_s = {'exploratory','confirmatory','analytical','comparative'}
        if method_type not in allowed_m:
            return None, ['method_type']
        if objective_scope not in allowed_s:
            return None, ['objective_scope']
        validation = extracted.get('validation_requirements', [])
        if not isinstance(validation, list) or not all(isinstance(v, str) and len(v.strip()) > 2 for v in validation):
            return None, ['validation_requirements']

        return (domain, keywords, method, objective, data_types, method_type, objective_scope, validation), []

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
) -> bool:
    raw = fullpaper.get("raw_output", "")
    # Ensure we interpret sections in order:
    headings = list(fullpaper.get("sections", {}).keys())

    errors = []

    # 1) Required headings
    REQUIRED = ["Abstract","Introduction","Literature Review",
                "Methodology","Experiments / Results","Conclusion"]
    missing_secs = [sec for sec in REQUIRED if sec not in headings]
    if missing_secs:
        errors.append(f"Missing sections: {missing_secs}")

    # 2) Parse into paragraphs
    sections = {}
    for i, sec in enumerate(headings):
        nxt = headings[i+1] if i+1 < len(headings) else None
        if nxt:
            pat = rf"(?:^|\n){re.escape(sec)}\n(.*?)(?=\n{re.escape(nxt)}\n)"
        else:
            pat = rf"(?:^|\n){re.escape(sec)}\n(.*)$"
        m = re.search(pat, raw, re.DOTALL)
        text = m.group(1).strip() if m else ""
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        sections[sec] = paras

    # 3) Paragraph length
    for sec, paras in sections.items():
        if not paras or all(len(p) < 200 for p in paras):
            errors.append(f"Section '{sec}' has no paragraph ≥200 chars.")

    # 4) Build citation maps
    bracket_map = {str(i+1): c for i,c in enumerate(citations)}
    authoryear_map = {}
    for c in citations:
        surname = c['author_names'].split()[0].lower()
        m = re.search(r",\s*(\d{4}|n\.d\.)", c['citation'])
        year = m.group(1).lower() if m else None
        if surname and year:
            authoryear_map[(surname, year)] = c

    summary_texts = [c["summary"].lower() for c in citations]

    # 5) Validate citations & factual claims
    for sec, paras in sections.items():
        for p in paras:
            # a) [n] citations
            for bid in re.findall(r"\[(\d+)\]", p):
                if bid not in bracket_map:
                    errors.append(f"Invalid bracket citation [{bid}] in '{sec}'.")

            # b) (Author, Year) citations
            for author_raw, year in re.findall(r"\(([^,]+),\s*(\d{4}|n\.d\.)\)", p):
                # skip 'n.d.'
                if year.lower() != "n.d.":
                    surname = re.sub(r"\bet\.?\s*al\.?", "", author_raw, flags=re.IGNORECASE).strip().lower()
                    if (surname, year.lower()) not in authoryear_map:
                        errors.append(f"Unknown citation ({author_raw}, {year}) in '{sec}'.")

            # c) factual SVO checks
            for sent in re.split(r"(?<=[\.\!?])\s+", p):
                if len(sent) < 20 or not re.search(r"\d|study|found|shows?|demonstrates?", sent, re.IGNORECASE):
                    continue
                words = sent.split()
                subj, verb, obj = words[0], words[1] if len(words)>1 else "", words[-1]
                supported = any(
                    verb.lower() in txt or obj.lower() in txt
                    for txt in summary_texts
                )
                if not supported:
                    errors.append(
                        f"Unsupported claim in '{sec}': “{subj} {verb} {obj}” not in any citation summary."
                    )

    # 6) Final outcome
    if errors:
        return False, errors

    return True, ""

def paper_rectification(
    fullpaper: Dict[str, any],
    citations: List[Dict[str, str]],
    errors: List[str],
    validation_requirements: List[str]
) -> Dict[str, any]:
    """
    Sends the original paper, citations, and validation errors plus requirements to the LLM for rectification.
    Returns a new fullpaper dict with 'raw_output' and 'sections'.
    """
    # 1) Build citation list string, one per line
    cit_lines = []
    for i, c in enumerate(citations, start=1):
        venue = c.get('venue', '')
        year  = c.get('year', '')
        cit = c.get('citation','')
        content = c.get('summary','')
        cit_lines.append((f"Citation : {cit}" if cit else "") + (f" -> Summary : {content}" if content else ""))
        # f"[{i}] {c['author_names']}, '{c['title']}'"
        #                  + (f", {venue}" if venue else "")
        #                  + (f", {year}"  if year else "")
    cit_text = "\n".join(cit_lines)

    # 2) Build the rectification prompt with clear separators and newlines
    rect_prompt = "\n".join([
        "Your previously generated research paper did not pass validation.",
        "Please revise it to address the following errors and adhere to these requirements. ",
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
        "Citations (use these exactly):",
        cit_text,
        "",
        "Produce a corrected full paper in plain text with the same six section headings:",
        "Abstract, Introduction, Literature Review, Methodology, Experiments / Results, Conclusion.",
        "Ensure all citations are properly formatted and that every factual claim is supported by one of them.",
        "Don't make too many changes. Make minimal to no changes to the already correct sentences. just change the lines that are mentioned in the error section (by either removing them altogether or by using an actial citation listed) also u have to make sure that the final draft fulfills all the validation requirements",
        "Reply with only providing the content and don't add any extra words (e.g here is what u asked for etc)"
    ])

    # 3) Call the LLM
    response = model.generate_content(rect_prompt, generation_config={"max_output_tokens": 16384})
    new_raw = response.text if getattr(response, "parts", None) else ""

    # 4) Parse sections exactly as in writing_agent
    section_titles = [
        "Abstract", "Introduction", "Literature Review",
        "Methodology", "Experiments / Results", "Conclusion"
    ]
    # Use a regex that splits on "\nSectionTitle\n"
    pattern = r"(?:^|\n)(" + "|".join(map(re.escape, section_titles)) + r")\n"
    splits = re.split(pattern, new_raw)
    new_sections = {}

    # splits: [ preamble, title1, content1, title2, content2, ... ]
    for i in range(1, len(splits), 2):
        title   = splits[i].strip()
        content = splits[i+1].strip()
        # paragraphs separated by two or more newlines
        paras = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
        new_sections[title] = paras if paras else [content]

    # Ensure all six sections exist
    for sec in section_titles:
        if sec not in new_sections:
            new_sections[sec] = ["This section was not generated."]

    return {"raw_output": new_raw, "sections": new_sections}


def paper_score(prompt, fullpaper: Dict[str, any]):

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2') 
    prompt_emb = model.encode([prompt])
    content_emb = model.encode([fullpaper['raw_output']])

    return cosine_similarity(prompt_emb, content_emb)[0][0]

