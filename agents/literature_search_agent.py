"""
Simple literature search agent for retrieving citations from arXiv.

This agent is a lightweight implementation that queries the arXiv API
and saves results to `docs/references.md`. It requires network access
and the `requests` package.

Usage:
    python agents/literature_search_agent.py --query "time series forecasting retail"

Note: This script is provided as an agent skeleton and can be extended to
call other APIs (CrossRef, Semantic Scholar) or to integrate with the project's
coordination/memory mechanism.
"""

import argparse
import requests
from xml.etree import ElementTree as ET
from pathlib import Path


def search_arxiv(query: str, max_results: int = 5):
    url = 'http://export.arxiv.org/api/query'
    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.text


def parse_arxiv(xml_text: str):
    root = ET.fromstring(xml_text)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    entries = []
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.strip()
        summary = entry.find('atom:summary', ns).text.strip()
        authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
        link = entry.find('atom:id', ns).text
        entries.append({'title': title, 'summary': summary, 'authors': authors, 'link': link})
    return entries


def save_references(entries, out_path='docs/references.md'):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        f.write('# References (arXiv search results)\n\n')
        for e in entries:
            f.write(f"## {e['title']}\n")
            f.write(f"**Authors**: {', '.join(e['authors'])}\n\n")
            f.write(f"**Link**: {e['link']}\n\n")
            f.write(f"{e['summary']}\n\n---\n\n")
    print(f"Saved {len(entries)} references to {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, required=True, help='Search query for arXiv')
    parser.add_argument('--max', type=int, default=5, help='Max results')
    args = parser.parse_args()

    print(f"Searching arXiv for: {args.query} (max {args.max})")
    xml = search_arxiv(args.query, max_results=args.max)
    entries = parse_arxiv(xml)
    save_references(entries)


if __name__ == '__main__':
    main()
