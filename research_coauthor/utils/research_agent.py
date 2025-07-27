import requests
import urllib.parse
import os
import time
import xml.etree.ElementTree as ET

API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

def get_crossref_summaries(keywords, max_results=2):
    query = ' '.join(keywords[:4])
    url = f'https://api.crossref.org/works?query={urllib.parse.quote(query)}&rows={max_results}'
    print(f"[DEBUG] Querying CrossRef with: {query}")
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            print(f"[DEBUG] CrossRef API returned status {response.status_code}: {response.text}")
            return []
        data = response.json()
        items = data.get('message', {}).get('items', [])
        results = []
        for i, item in enumerate(items):
            title = item.get('title', ['No Title'])[0]
            authors = item.get('author', [])
            print(f"[DEBUG] CrossRef Paper {i+1} raw authors: {authors}")
            author_names = ', '.join([f"{a.get('given', '')} {a.get('family', '')}".strip() for a in authors[:3]]) or "Unknown Author"
            print(f"[DEBUG] CrossRef Paper {i+1} extracted author_names: '{author_names}'")
            year = item.get('issued', {}).get('date-parts', [["n.d."]])[0][0]
            venue = item.get('container-title', ['Unknown Venue'])[0]
            abstract = item.get('abstract', '')
            citation = f"[{i+1}] {author_names}, '{title}', {venue}, {year}"
            summary = abstract if abstract else f"{title} discusses topics related to {query}."
            results.append({
                'title': title,
                'summary': summary,
                'citation': citation,
                'ref': citation,
                'author_names': author_names
            })
            print(f"[DEBUG] CrossRef Paper {i+1}: {title} | Authors: {author_names}")
        return results
    except Exception as e:
        print(f"[DEBUG] Exception in get_crossref_summaries: {e}")
        return []

def get_arxiv_summaries(keywords, max_results=2):
    query = '+'.join(keywords[:4])
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}'
    print(f"[DEBUG] Querying arXiv with: {query}")
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            print(f"[DEBUG] arXiv API returned status {response.status_code}: {response.text}")
            return []
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        results = []
        for i, entry in enumerate(entries):
            title_elem = entry.find('atom:title', ns)
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else 'No Title'
            authors = entry.findall('atom:author/atom:name', ns)
            author_names = ', '.join([a.text for a in authors[:3] if a is not None and a.text]) or "Unknown Author"
            year_elem = entry.find('atom:published', ns)
            year = year_elem.text[:4] if year_elem is not None and year_elem.text else 'n.d.'
            summary_elem = entry.find('atom:summary', ns)
            summary = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else f"{title} discusses topics related to {query}."
            citation = f"[{i+1}] {author_names}, '{title}', arXiv, {year}"
            results.append({
                'title': title,
                'summary': summary,
                'citation': citation,
                'ref': citation,
                'author_names': author_names
            })
            print(f"[DEBUG] arXiv Paper {i+1}: {title} | Authors: {author_names}")
        return results
    except Exception as e:
        print(f"[DEBUG] Exception in get_arxiv_summaries: {e}")
        return []

def get_real_source_summaries(keywords, max_results=2):
    """
    Query Semantic Scholar API for real academic papers based on keywords.
    Returns a list of dicts with title, summary, citation, ref, and author_names.
    Uses only the full keyword list as a single query. Handles rate limits and network errors gracefully.
    Filters results for relevance using only LLM-extracted keywords.
    Fallbacks to CrossRef and arXiv if no results.
    """
    headers = {}
    if API_KEY:
        headers['x-api-key'] = API_KEY
    if isinstance(keywords, str):
        keywords = [keywords]
    elif not isinstance(keywords, list):
        keywords = list(keywords)
    cleaned_keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
    print(f"[DEBUG] Final cleaned keywords for search: {cleaned_keywords}")
    if not cleaned_keywords:
        print("[DEBUG] No valid keywords provided. Returning placeholder result.")
        return [{
            'title': 'No relevant papers found',
            'summary': 'No real literature found for these keywords.',
            'citation': '[1] Placeholder Author, "No relevant papers found", Placeholder Journal, 2024',
            'ref': '[1] Placeholder Author, "No relevant papers found", Placeholder Journal, 2024',
            'author_names': 'Unknown Author'
        }]
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
            return get_crossref_summaries(cleaned_keywords, max_results) or get_arxiv_summaries(cleaned_keywords, max_results)
        if response.status_code != 200:
            print(f"[DEBUG] API returned status {response.status_code}: {response.text}")
            return get_crossref_summaries(cleaned_keywords, max_results) or get_arxiv_summaries(cleaned_keywords, max_results)
        data = response.json()
        print(f"[DEBUG] API Response keys: {list(data.keys())}")
        papers = data.get('data', [])
        print(f"[DEBUG] Found {len(papers)} papers for query '{query}'")
        # Filter for relevance using only LLM-extracted keywords
        if cleaned_keywords:
            filter_terms = [k.lower() for k in cleaned_keywords if isinstance(k, str)]
            filtered_papers = []
            for paper in papers:
                # Fix the concatenation error by handling None values properly
                title = paper.get('title') or ''
                abstract = paper.get('abstract') or ''
                # Ensure both are strings before concatenation
                title = str(title) if title is not None else ''
                abstract = str(abstract) if abstract is not None else ''
                text = (title + ' ' + abstract).lower()
                if any(term in text for term in filter_terms):
                    filtered_papers.append(paper)
            if not filtered_papers:
                print("[DEBUG] No relevant papers found after filtering for LLM-extracted keywords. Trying CrossRef and arXiv.")
                return get_crossref_summaries(cleaned_keywords, max_results) or get_arxiv_summaries(cleaned_keywords, max_results)
            papers = filtered_papers
        results = []
        for i, paper in enumerate(papers):
            title = paper.get('title', 'No Title')
            authors = paper.get('authors', [])
            print(f"[DEBUG] Paper {i+1} raw authors data: {authors} (type: {type(authors)})")
            print(f"[DEBUG] Paper {i+1} full paper data keys: {list(paper.keys())}")
            # Better author extraction
            author_list = []
            for a in authors[:3]:
                name = a.get('name', '').strip()
                if name:  # Only add non-empty names
                    author_list.append(name)
            # If no authors found, try alternative fields
            if not author_list:
                print(f"[DEBUG] Paper {i+1} no authors in 'authors' field, trying alternatives...")
                # Try different possible author fields
                for field in ['author', 'creator', 'contributor']:
                    alt_authors = paper.get(field, [])
                    if alt_authors:
                        print(f"[DEBUG] Paper {i+1} found authors in '{field}': {alt_authors}")
                        for a in alt_authors[:3]:
                            if isinstance(a, dict):
                                name = a.get('name', '').strip()
                            else:
                                name = str(a).strip()
                            if name:
                                author_list.append(name)
                        if author_list:
                            break
            author_names = ', '.join(author_list) if author_list else "Unknown Author"
            print(f"[DEBUG] Paper {i+1} extracted author_names: '{author_names}'")
            # Better year extraction with debugging
            raw_year = paper.get('year')
            print(f"[DEBUG] Paper {i+1} raw year data: {raw_year} (type: {type(raw_year)})")
            if raw_year is not None and raw_year != '':
                year = str(raw_year)
            else:
                year = 'n.d.'
            print(f"[DEBUG] Paper {i+1} final year: '{year}'")
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
        if not results:
            print("[DEBUG] No results from Semantic Scholar, trying CrossRef and arXiv.")
            return get_crossref_summaries(cleaned_keywords, max_results) or get_arxiv_summaries(cleaned_keywords, max_results)
        return results
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] Request exception: {e}")
        return get_crossref_summaries(cleaned_keywords, max_results) or get_arxiv_summaries(cleaned_keywords, max_results)
    except Exception as e:
        print(f"[DEBUG] Exception in get_real_source_summaries: {e}")
        return get_crossref_summaries(cleaned_keywords, max_results) or get_arxiv_summaries(cleaned_keywords, max_results)

# Robust summary extraction for research papers