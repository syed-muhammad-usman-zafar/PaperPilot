import requests
import urllib.parse
import os
import time

API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

def get_real_source_summaries(keywords, max_results=2):
    """
    Query Semantic Scholar API for real academic papers based on keywords.
    Returns a list of dicts with title, summary, citation, ref, and author_names.
    Uses only the full keyword list as a single query. Handles rate limits and network errors gracefully.
    Filters results for relevance using only LLM-extracted keywords.
    """
    headers = {}
    if API_KEY:
        headers['x-api-key'] = API_KEY
    if isinstance(keywords, str):
        keywords = [keywords]
    elif not isinstance(keywords, list):
        keywords = list(keywords)
    cleaned_keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
    if not cleaned_keywords:
        print("[DEBUG] No valid keywords provided, using fallback 'machine learning'")
        cleaned_keywords = ["machine learning"]
    query = ' '.join(cleaned_keywords[:4])
    encoded_query = urllib.parse.quote(query)
    url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&fields=title,authors,year,venue,abstract&limit={max_results}'
    print(f"[DEBUG] Querying Semantic Scholar with: {query}")
    print(f"[DEBUG] API URL: {url}")
    try:
        response = requests.get(url, timeout=15, headers=headers)
        print(f"[DEBUG] Response status code: {response.status_code}")
        if response.status_code == 429:
            print(f"[DEBUG] API returned status 429 (rate limit): {response.text}")
            print("[DEBUG] User warning: Too many requests to Semantic Scholar API. Please wait or use an API key.")
            return []
        if response.status_code != 200:
            print(f"[DEBUG] API returned status {response.status_code}: {response.text}")
            return []
        data = response.json()
        print(f"[DEBUG] API Response keys: {list(data.keys())}")
        papers = data.get('data', [])
        print(f"[DEBUG] Found {len(papers)} papers for query '{query}'")
        # Filter for relevance using only LLM-extracted keywords
        if cleaned_keywords:
            filter_terms = [k.lower() for k in cleaned_keywords if isinstance(k, str)]
            filtered_papers = []
            for paper in papers:
                text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
                if any(term in text for term in filter_terms):
                    filtered_papers.append(paper)
            if not filtered_papers:
                print("[DEBUG] No relevant papers found after filtering for LLM-extracted keywords. Returning empty list.")
                return []
            papers = filtered_papers
        results = []
        for i, paper in enumerate(papers):
            title = paper.get('title', 'No Title')
            authors = paper.get('authors', [])
            author_names = ', '.join([a.get('name', '') for a in authors[:3]]) or "Unknown Author"
            year = paper.get('year', 'n.d.')
            venue = paper.get('venue', 'Unknown Venue')
            abstract = paper.get('abstract', '')
            citation = f"[{i+1}] {author_names}, '{title}', {venue}, {year}"
            summary = abstract if abstract else f"{title} discusses topics related to {query}."
            results.append({
                'title': title,
                'summary': summary,
                'citation': citation,
                'ref': citation,
                'author_names': author_names
            })
            print(f"[DEBUG] Paper {i+1}: {title} | Authors: {author_names}")
        return results
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] Request exception: {e}")
        return []
    except Exception as e:
        print(f"[DEBUG] Exception in get_real_source_summaries: {e}")
        return [] 