import json
import re
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict


# Stopwords from NLTK
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'am', 'be', 'been', 'being',
    'that', 'this', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'as', 'was', 'were', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'same', 'so', 'than', 'too', 'very', 'just', 'my',
}


@dataclass
class DocumentMetadata:
    """Metadata for a document."""
    url: str
    title: str
    description: str
    brand: Optional[str] = None
    origin: Optional[str] = None
    total_reviews: int = 0
    mean_rating: float = 0.0


@dataclass
class SearchResult:
    """Search result with ranking information."""
    url: str
    title: str
    description: str
    score: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'url': self.url,
            'title': self.title,
            'description': self.description,
            'score': round(self.score, 4),
            'metadata': self.metadata
        }


class Tokenizer:
    """Handles tokenization of text."""
    
    def __init__(self, stopwords: Set[str] = None):
        """
        Initialize tokenizer.
        
        Args:
            stopwords: Set of stopwords to remove
        """
        self.stopwords = stopwords or STOPWORDS
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text by removing punctuation and stopwords.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        # Keep only alphanumeric characters and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Split by spaces
        tokens = text.split()
        # Remove stopwords and empty strings
        tokens = [t for t in tokens if t and t not in self.stopwords]
        return tokens
    
    def tokenize_without_stopwords(self, text: str) -> List[str]:
        """Tokenize without removing stopwords (for matching)."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if t]


class SynonymProcessor:
    """Handles synonym expansion for queries."""
    
    def __init__(self, synonyms_file: str):
        """
        Initialize synonym processor.
        
        Args:
            synonyms_file: Path to JSON synonyms file
        """
        self.synonyms = self._load_synonyms(synonyms_file)
    
    def _load_synonyms(self, file_path: str) -> Dict[str, List[str]]:
        """Load synonyms from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Synonyms file not found: {file_path}")
            return {}
    
    def expand_query(self, tokens: List[str]) -> List[str]:
        """
        Expand query tokens with synonyms.
        
        Args:
            tokens: Original query tokens
            
        Returns:
            Expanded list of tokens including synonyms
        """
        expanded = list(tokens)
        for token in tokens:
            if token in self.synonyms:
                # Add all synonyms
                synonyms_list = self.synonyms[token]
                tokenizer = Tokenizer()
                for synonym in synonyms_list:
                    expanded.extend(tokenizer.tokenize(synonym))
        return list(set(expanded))  # Remove duplicates


class IndexLoader:
    """Loads and manages inverted indexes."""
    
    def __init__(self, indexes_dir: str):
        """
        Initialize index loader.
        
        Args:
            indexes_dir: Directory containing index files
        """
        self.indexes_dir = Path(indexes_dir)
        self.title_index = {}
        self.description_index = {}
        self.brand_index = {}
        self.origin_index = {}
        self.reviews_index = {}
        self.documents = {}  # URL -> {title, description}
    
    def load_all_indexes(self):
        """Load all available indexes."""
        print("Loading indexes...")
        self.title_index = self._load_index('title_index.json')
        self.description_index = self._load_index('description_index.json')
        self.brand_index = self._load_index('brand_index.json')
        self.origin_index = self._load_index('origin_index.json')
        self.reviews_index = self._load_index('reviews_index.json')
        print(f"[OK] Loaded {len(self.title_index)} tokens from title index")
        print(f"[OK] Loaded {len(self.description_index)} tokens from description index")
    
    def _load_index(self, filename: str) -> Dict:
        """Load a single index file."""
        file_path = self.indexes_dir / filename
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Index file not found: {file_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {file_path}")
            return {}
    
    def build_document_cache(self, metadata_file: Optional[str] = None):
        """
        Build cache of document metadata.
        
        Args:
            metadata_file: Optional path to metadata JSON file
        """
        # Extract documents from title index
        for token, urls_dict in self.title_index.items():
            for url in urls_dict.keys():
                if url not in self.documents:
                    self.documents[url] = {
                        'title': '',
                        'description': '',
                        'token_count': 0
                    }


class DocumentFilter:
    """Handles document filtering based on query tokens."""
    
    def __init__(self, tokenizer: Tokenizer, synonym_processor: SynonymProcessor):
        """
        Initialize filter.
        
        Args:
            tokenizer: Tokenizer instance
            synonym_processor: SynonymProcessor instance
        """
        self.tokenizer = tokenizer
        self.synonym_processor = synonym_processor
    
    def prepare_query(self, query: str, expand_synonyms: bool = False) -> List[str]:
        """
        Prepare query by tokenizing and optionally expanding synonyms.
        
        Args:
            query: Raw query string
            expand_synonyms: Whether to expand with synonyms
            
        Returns:
            List of query tokens
        """
        tokens = self.tokenizer.tokenize(query)
        if expand_synonyms:
            tokens = self.synonym_processor.expand_query(tokens)
        return tokens
    
    def filter_with_any_token(self, query_tokens: List[str], 
                             title_index: Dict, description_index: Dict,
                             brand_index: Dict, origin_index: Dict) -> Set[str]:
        """
        Filter documents that contain at least one query token.
        
        Args:
            query_tokens: Tokenized query
            title_index: Title inverted index
            description_index: Description inverted index
            brand_index: Brand inverted index
            origin_index: Origin inverted index
            
        Returns:
            Set of matching document URLs
        """
        results = set()
        for token in query_tokens:
            # Check title
            if token in title_index:
                results.update(title_index[token].keys())
            # Check description
            if token in description_index:
                results.update(description_index[token].keys())
            # Check brand
            if token in brand_index:
                results.update(brand_index[token])
            # Check origin
            if token in origin_index:
                results.update(origin_index[token])
        return results
    
    def filter_with_all_tokens(self, query_tokens: List[str],
                              title_index: Dict, description_index: Dict,
                              brand_index: Dict, origin_index: Dict) -> Set[str]:
        """
        Filter documents that contain ALL tokens (except stopwords).
        
        Args:
            query_tokens: Tokenized query
            title_index: Title inverted index
            description_index: Description inverted index
            brand_index: Brand inverted index
            origin_index: Origin inverted index
            
        Returns:
            Set of matching document URLs
        """
        # Separate stopwords from regular tokens
        non_stopword_tokens = [t for t in query_tokens if t not in STOPWORDS]
        
        if not non_stopword_tokens:
            # If only stopwords, match any
            return self.filter_with_any_token(query_tokens, title_index,
                                             description_index, brand_index, origin_index)
        
        # Start with first token
        first_token = non_stopword_tokens[0]
        results = set()
        
        # Collect all URLs for first token
        if first_token in title_index:
            results.update(title_index[first_token].keys())
        if first_token in description_index:
            results.update(description_index[first_token].keys())
        if first_token in brand_index:
            results.update(brand_index[first_token])
        if first_token in origin_index:
            results.update(origin_index[first_token])
        
        # Filter by remaining tokens
        for token in non_stopword_tokens[1:]:
            token_urls = set()
            if token in title_index:
                token_urls.update(title_index[token].keys())
            if token in description_index:
                token_urls.update(description_index[token].keys())
            if token in brand_index:
                token_urls.update(brand_index[token])
            if token in origin_index:
                token_urls.update(origin_index[token])
            
            results = results.intersection(token_urls)
            if not results:
                break
        
        return results


class BM25Scorer:
    """BM25 ranking algorithm."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 scorer.
        
        Args:
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self.k1 = k1
        self.b = b
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.num_docs = 0
        self.idf = {}
    
    def fit(self, title_index: Dict, description_index: Dict):
        """
        Fit BM25 parameters on the corpus.
        
        Args:
            title_index: Title inverted index
            description_index: Description inverted index
        """
        # Collect all document URLs
        all_urls = set()
        for urls_dict in title_index.values():
            all_urls.update(urls_dict.keys())
        for urls_dict in description_index.values():
            all_urls.update(urls_dict.keys())
        
        self.num_docs = len(all_urls)
        
        # Calculate IDF for all tokens
        combined_index = {}
        for token, urls_dict in title_index.items():
            if token not in combined_index:
                combined_index[token] = set()
            combined_index[token].update(urls_dict.keys())
        
        for token, urls_dict in description_index.items():
            if token not in combined_index:
                combined_index[token] = set()
            combined_index[token].update(urls_dict.keys())
        
        # Calculate IDF
        for token, urls in combined_index.items():
            doc_freq = len(urls)
            self.idf[token] = math.log(
                (self.num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0
            )
    
    def score_document(self, tokens: List[str], token_frequencies: Dict[str, int],
                      doc_length: int) -> float:
        """
        Score a document using BM25.
        
        Args:
            tokens: Query tokens
            token_frequencies: Token frequencies in document
            doc_length: Length of document (token count)
            
        Returns:
            BM25 score
        """
        score = 0.0
        
        for token in tokens:
            if token not in token_frequencies:
                continue
            
            freq = token_frequencies[token]
            idf = self.idf.get(token, 0)
            
            # BM25 formula
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * (doc_length / max(self.avg_doc_length, 1)))
            score += idf * (numerator / denominator)
        
        return score


class PositionScorer:
    """Scorer based on token positions."""
    
    @staticmethod
    def calculate_position_score(positions: List[int]) -> float:
        """
        Calculate score based on positions.
        Earlier positions get higher scores.
        
        Args:
            positions: List of positions where token appears
            
        Returns:
            Position-based score
        """
        if not positions:
            return 0.0
        
        # First occurrence position
        first_pos = min(positions)
        
        # Decay based on position (earlier = higher)
        # Score decreases as position increases
        position_score = 1.0 / (1.0 + first_pos * 0.1)
        
        # Bonus for multiple occurrences
        occurrence_bonus = math.log(len(positions) + 1) * 0.1
        
        return position_score + occurrence_bonus


class LinearRanker:
    """Linear combination ranking combining multiple signals."""
    
    def __init__(self):
        """Initialize ranker with weights."""
        # Weights for different signals
        self.weights = {
            'title_presence': 2.0,        # Presence in title
            'title_bm25': 2.5,             # BM25 score in title
            'title_position': 0.3,         # Early position in title
            'description_bm25': 1.0,       # BM25 score in description
            'description_position': 0.2,   # Early position in description
            'exact_match': 1.5,            # Exact phrase match
            'frequency': 0.5,              # Token frequency
            'reviews': 0.8,                # Review signals
            'review_recency': 0.3,         # Recent reviews bonus
        }
    
    def rank(self, url: str, query_tokens: List[str],
            title_text: str, description_text: str,
            title_index: Dict, description_index: Dict,
            reviews_data: Dict, tokenizer: Tokenizer,
            bm25_scorer: BM25Scorer) -> Dict[str, float]:
        """
        Calculate ranking score for a document.
        
        Args:
            url: Document URL
            query_tokens: Query tokens
            title_text: Document title
            description_text: Document description
            title_index: Title inverted index
            description_index: Description inverted index
            reviews_data: Reviews data for document
            tokenizer: Tokenizer instance
            bm25_scorer: BM25 scorer instance
            
        Returns:
            Dictionary with detailed scores
        """
        scores = {
            'title_presence': 0.0,
            'title_bm25': 0.0,
            'title_position': 0.0,
            'description_bm25': 0.0,
            'description_position': 0.0,
            'exact_match': 0.0,
            'frequency': 0.0,
            'reviews': 0.0,
            'review_recency': 0.0,
            'total': 0.0
        }
        
        # Tokenize title and description
        title_tokens = tokenizer.tokenize_without_stopwords(title_text)
        desc_tokens = tokenizer.tokenize_without_stopwords(description_text)
        
        # 1. Title signals
        matching_title_tokens = set(query_tokens) & set(title_tokens)
        title_freqs = {}
        for token in query_tokens:
            title_freqs[token] = title_tokens.count(token)
        
        if matching_title_tokens:
            scores['title_presence'] = len(matching_title_tokens)
            
            # BM25 score in title
            scores['title_bm25'] = bm25_scorer.score_document(
                query_tokens, title_freqs, len(title_tokens)
            )
            
            # Position score in title
            for token in matching_title_tokens:
                positions = [i for i, t in enumerate(title_tokens) if t == token]
                scores['title_position'] += PositionScorer.calculate_position_score(positions)
        
        # 2. Description signals
        matching_desc_tokens = set(query_tokens) & set(desc_tokens)
        if matching_desc_tokens:
            # BM25 score in description
            desc_freqs = {}
            for token in query_tokens:
                desc_freqs[token] = desc_tokens.count(token)
            scores['description_bm25'] = bm25_scorer.score_document(
                query_tokens, desc_freqs, len(desc_tokens)
            )
            
            # Position score in description
            for token in matching_desc_tokens:
                positions = [i for i, t in enumerate(desc_tokens) if t == token]
                scores['description_position'] += PositionScorer.calculate_position_score(positions)
        
        # 3. Exact phrase match
        query_phrase = ' '.join(query_tokens)
        title_norm = ' '.join(title_tokens)
        desc_norm = ' '.join(desc_tokens)
        if query_phrase in title_norm:
            scores['exact_match'] = 2.0
        elif query_phrase in desc_norm:
            scores['exact_match'] = 1.0
        
        # 4. Token frequency
        total_freq = title_freqs.get(query_tokens[0], 0) if query_tokens else 0
        if query_tokens and len(query_tokens) > 0:
            for token in query_tokens:
                total_freq += title_tokens.count(token) + desc_tokens.count(token)
        scores['frequency'] = min(total_freq * 0.2, 1.0)  # Cap at 1.0
        
        # 5. Review signals
        if url in reviews_data:
            review_info = reviews_data[url]
            total_reviews = review_info.get('total_reviews', 0)
            mean_mark = review_info.get('mean_mark', 0)
            
            # Normalize review count (0-1 scale, capped at 20 reviews)
            reviews_score = min(total_reviews / 20.0, 1.0)
            scores['reviews'] = reviews_score * (mean_mark / 5.0)  # Weight by rating
            
            # Recent review bonus
            last_rating = review_info.get('last_rating', 0)
            if last_rating >= 4:  # Recent positive review
                scores['review_recency'] = 0.3
        
        # Calculate weighted total
        total_score = 0.0
        for signal, weight in self.weights.items():
            if signal in scores:
                total_score += scores[signal] * weight
        
        scores['total'] = total_score
        return scores


class SearchEngine:
    """Main search engine class."""
    
    def __init__(self, indexes_dir: str = 'inputs'):
        """
        Initialize search engine.
        
        Args:
            indexes_dir: Directory containing index files
        """
        self.indexes_dir = Path(indexes_dir)
        self.tokenizer = Tokenizer()
        self.synonym_processor = SynonymProcessor(str(self.indexes_dir / 'origin_synonyms.json'))
        self.index_loader = IndexLoader(str(self.indexes_dir))
        self.document_filter = DocumentFilter(self.tokenizer, self.synonym_processor)
        self.bm25_scorer = BM25Scorer()
        self.linear_ranker = LinearRanker()
        
        # Load all data
        self._initialize()
    
    def _initialize(self):
        """Initialize engine by loading indexes."""
        self.index_loader.load_all_indexes()
        self.index_loader.build_document_cache()
        self.bm25_scorer.fit(
            self.index_loader.title_index,
            self.index_loader.description_index
        )
        print(f"[OK] Search engine initialized with {self.bm25_scorer.num_docs} documents")
    
    def search(self, query: str, filter_mode: str = 'any',
              expand_synonyms: bool = False, top_k: int = 10) -> Dict[str, Any]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query string
            filter_mode: 'any' to match any token, 'all' to match all tokens
            expand_synonyms: Whether to expand query with synonyms
            top_k: Number of top results to return
            
        Returns:
            Dictionary with search results and metadata
        """
        # Prepare query
        query_tokens = self.document_filter.prepare_query(query, expand_synonyms)
        
        if not query_tokens:
            return {
                'query': query,
                'filter_mode': filter_mode,
                'total_documents': len(self.index_loader.documents),
                'filtered_count': 0,
                'results': [],
                'query_tokens': []
            }
        
        # Filter documents
        if filter_mode == 'any':
            matching_urls = self.document_filter.filter_with_any_token(
                query_tokens,
                self.index_loader.title_index,
                self.index_loader.description_index,
                self.index_loader.brand_index,
                self.index_loader.origin_index
            )
        else:  # 'all'
            matching_urls = self.document_filter.filter_with_all_tokens(
                query_tokens,
                self.index_loader.title_index,
                self.index_loader.description_index,
                self.index_loader.brand_index,
                self.index_loader.origin_index
            )
        
        # Rank documents
        scored_results = []
        for url in matching_urls:
            title = self._get_document_title(url)
            description = self._get_document_description(url)
            
            # Get detailed scores
            detailed_scores = self.linear_ranker.rank(
                url, query_tokens, title, description,
                self.index_loader.title_index,
                self.index_loader.description_index,
                self.index_loader.reviews_index,
                self.tokenizer,
                self.bm25_scorer
            )
            
            result = SearchResult(
                url=url,
                title=title,
                description=description,
                score=detailed_scores['total'],
                metadata={
                    'detailed_scores': {k: v for k, v in detailed_scores.items() if k != 'total'},
                    'reviews': self.index_loader.reviews_index.get(url, {})
                }
            )
            scored_results.append(result)
        
        # Sort by score and take top-k
        scored_results.sort(key=lambda x: x.score, reverse=True)
        top_results = scored_results[:top_k]
        
        return {
            'query': query,
            'query_tokens': query_tokens,
            'filter_mode': filter_mode,
            'expand_synonyms': expand_synonyms,
            'total_documents': len(self.index_loader.documents),
            'filtered_count': len(matching_urls),
            'returned_count': len(top_results),
            'results': [r.to_dict() for r in top_results]
        }
    
    def _get_document_title(self, url: str) -> str:
        """Extract title from title index positions."""
        # Try to reconstruct from index
        title_tokens = []
        for token, urls_dict in self.index_loader.title_index.items():
            if url in urls_dict:
                positions_list = urls_dict[url]
                title_tokens.append((token, min(positions_list)))
        
        title_tokens.sort(key=lambda x: x[1])
        title = ' '.join([t[0] for t in title_tokens])
        
        return title or f"Document: {url}"
    
    def _get_document_description(self, url: str) -> str:
        """Extract description from description index."""
        # Try to reconstruct from index
        desc_tokens = []
        for token, urls_dict in self.index_loader.description_index.items():
            if url in urls_dict:
                positions_list = urls_dict[url]
                desc_tokens.append((token, min(positions_list)))
        
        desc_tokens.sort(key=lambda x: x[1])
        description = ' '.join([t[0] for t in desc_tokens[:50]])  # First 50 tokens
        
        return description or "No description available"
    
    def batch_search(self, queries: List[str], filter_mode: str = 'any',
                    expand_synonyms: bool = False, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform multiple searches.
        
        Args:
            queries: List of query strings
            filter_mode: Filter mode for all queries
            expand_synonyms: Whether to expand with synonyms
            top_k: Number of results per query
            
        Returns:
            List of search results
        """
        results = []
        for query in queries:
            result = self.search(query, filter_mode, expand_synonyms, top_k)
            results.append(result)
        return results
    
    def export_results(self, search_results: Dict[str, Any], 
                      output_file: str) -> None:
        """
        Export search results to JSON file.
        
        Args:
            search_results: Results from search method
            output_file: Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(search_results, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] Results exported to {output_file}")


def main():
    """Main function for testing."""
    print("Initializing Search Engine")
    
    engine = SearchEngine('inputs')
    
    # Test queries
    test_queries = [
        "laptop computer",
        "blue shirt",
        "made in france",
        "high quality product",
        "excellent reviews",
    ]
    
    print("\nPerforming test searches...")
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = engine.search(query, filter_mode='any', top_k=5)
        
        print(f"Filtered: {results['filtered_count']} documents")
        print(f"Returned: {results['returned_count']} results")
        
        for i, result in enumerate(results['results'], 1):
            print(f"  {i}. Score: {result['score']:.4f}")
            print(f"     Title: {result['title'][:60]}...")


if __name__ == '__main__':
    main()
