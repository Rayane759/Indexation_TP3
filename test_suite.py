from search_engine import SearchEngine


def run_tests():
    engine = SearchEngine('inputs')
    
    # Test queries
    test_queries = [
        ("blue", "any", False),
        ("waterproof", "any", False),
        ("usa made", "any", True),
        ("black white", "any", False),
        ("stylish modern", "any", False),
    ]
    
    for query, filter_mode, expand_synonyms in test_queries:
        print(f"\nQuery: '{query}'")
        print(f"Mode: {filter_mode} | Synonyms: {expand_synonyms}")
        
        results = engine.search(
            query,
            filter_mode=filter_mode,
            expand_synonyms=expand_synonyms,
            top_k=5
        )
        
        print(f"Filtered: {results['filtered_count']} | Results: {results['returned_count']}")
        
        if results['results']:
            print("Top result:")
            top = results['results'][0]
            print(f"  Score: {top['score']:.4f}")
            print(f"  Title: {top['title']}")
        else:
            print("No results found")


if __name__ == '__main__':
    run_tests()
