# Search Engine - Complete Implementation

## Overview

A search engine for product data combining BM25 ranking with 9 signals (token frequency, position, reviews). Implements dual-mode filtering, synonym expansion, and configurable weight optimization.

---

## Quick Start

```bash
# Run test suite
python test_suite.py

# Use the engine in Python
from search_engine import SearchEngine
engine = SearchEngine('inputs')
results = engine.search("blue", filter_mode='any', top_k=10)
```

---

## Architecture & Design Choices

### Multi-Signal Ranking System (9 Signals)

The engine combines 9 different ranking signals :

| # | Signal | Weight | Purpose |
|---|--------|--------|---------|
| 1 | Title Presence | 2.0 | Count of query tokens in title |
| 2 | Title BM25 | 2.5 | BM25 score on title (most important) |
| 3 | Title Position | 0.3 | Early positions score higher |
| 4 | Description BM25 | 1.0 | BM25 score on description |
| 5 | Description Position | 0.2 | Early positions in description |
| 6 | Exact Match | 1.5 | Bonus for exact phrase match |
| 7 | Token Frequency | 0.5 | Overall term count in document |
| 8 | Reviews | 0.8 | Review count * normalized rating |
| 9 | Review Recency | 0.3 | Bonus for recent positive reviews |

**Why this approach?**
- Single signals (only TF-IDF) are not enough
- Title is more important than description
- User reviews are quality indicators
- Early document positions reflect natural structure
- Exact phrases need explicit bonus

### BM25 Algorithm Implementation

```
BM25(q,d) = Sum IDF(term) * (freq * (k1+1)) / (freq + k1*(1-b+b*|d|/avgdl))
```

**Parameters chosen**:
- **k1 = 1.5**: Prevents term frequency saturation (avoids over-weighting long documents)
- **b = 0.75**: Balances length normalization (long documents get slight advantage if relevant)

**Why BM25?**
- Better than TF-IDF (accounts for document length)
- Proven effective in information retrieval
- Interpretable parameters
- Good balance between simplicity and effectiveness

### Position-Aware Scoring

```
position_score = 1/(1 + position*0.1) + log(occurrences+1)*0.1
```

**Why position matters**:
- Important information typically appears first
- Early occurrence = higher relevance signal
- Multiple occurrences get logarithmic bonus
- Reflects how humans write documents

### Review Signal Integration

- **Review count**: Normalized to 0-1 scale (cap at 20 reviews)
- **Rating impact**: Final score = (count/20) * (rating/5)
- **Recency bonus**: +0.3 if last review >= 4 stars
- **Rationale**: User feedback is quality indicator; recent reviews more valuable

### Filtering Strategies

**"Any Token" Mode** (Default):
- Matches documents containing >= 1 query term
- Higher recall, moderate precision
- Use for exploratory searches

**"All Token" Mode** (Strict):
- Matches documents with all non-stopword terms
- Lower recall, higher precision
- Use for specific intent searches

**Stopword Handling**:
- Standard English stopwords filtered
- All-token mode requires only non-stopwords
- Prevents common words from dominating

### Synonym Expansion

- Loaded from `origin_synonyms.json`
- Improves recall for location-based searches
- Optional per-query

### Default Weights Configuration

The engine uses a single optimized default weight configuration:
- **Title Presence**: 2.0 (count of query tokens)
- **Title BM25**: 2.5 (most important signal)
- **Title Position**: 0.3 (early positions)
- **Description BM25**: 1.0 (secondary field)
- **Description Position**: 0.2 (fine-tuning)
- **Exact Match**: 1.5 (phrase matching)
- **Frequency**: 0.5 (term repetition)
- **Reviews**: 0.8 (user quality signal)
- **Review Recency**: 0.3 (recent feedback)

Custom weights can be easily modified by changing `RankingWeights.DEFAULT` in `config.py`.

---

## Test Results Summary

### Corpus Statistics
- **Total documents**: 156 indexed
- **Unique tokens**: 332 (62 titles, 299 descriptions)
- **Review coverage**: 84.6% have reviews
- **Average rating**: 4.47/5 stars

### Test Suite Results (5 Core Queries)

| Query | Mode | Synonyms | Filtered | Results | Top Score | Status |
|-------|------|----------|----------|---------|-----------|--------|
| **blue** | any | No | 25 | 5 | 8.59 | Pass |
| **waterproof** | any | No | 7 | 5 | 2.28 | Pass |
| **usa made** | any | Yes | 50 | 5 | 0.80 | Pass |
| **black white** | any | No | 23 | 5 | 0.99 | Pass |
| **stylish modern** | any | No | 10 | 5 | 0.75 | Pass |

### Key Findings

**Strengths Demonstrated**:
- Synonym expansion effective ("usa made" -> 50 matches)
- Single-term searches work well ("blue" -> 8.59 score)
- Multi-word queries properly ranked
- Review signals properly integrated (highly-rated products rank higher)
- Score distribution reasonable (stdev 0.03-0.50)

**Observations**:
- All-token filter very strict (most queries return 0)
- Corpus has limited vocabulary overlap (many queries -> 0 results)
- Synonym expansion critical for origin searches
- Simple terms perform better than complex queries

---

## Usage Guide

```python
from search_engine import SearchEngine

engine = SearchEngine('inputs')

# Simple search
results = engine.search("blue", filter_mode='any', top_k=10)
for result in results['results']:
    print(f"{result['score']:.4f}: {result['title']}")
```

---

## Running Tests

```bash
# Run the test suite
python test_suite.py
```

Output shows test queries with filtering, ranking, and top results. All results are deterministic based on fixed index files.

---

## Summary

This search engine implements the 4-step information retrieval pipeline:

1. **Reading & Preparation**: Load and tokenize indexes
2. **Document Filtering**: Match documents by token (any/all modes)
3. **Ranking**: 9-signal linear combination with BM25 + position scoring
4. **Testing**: Validate with test queries

**Key features**:
- Multi-signal ranking (9 signals)
- BM25 with position-aware scoring
- Review quality integration
- Synonym expansion support
- Dual-mode filtering
- Production-ready code
