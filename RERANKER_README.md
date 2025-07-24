# Reranker Functionality Documentation

## Overview

The reranker functionality has been added to improve the relevance of retrieved documents in your RAG system. It uses advanced ML models (cross-encoders) to re-score and reorder documents based on their semantic relevance to the query.

## Components Added

### 1. `reranker.py` - Core Reranking Module

Contains three main classes:

#### DocumentReranker
- Uses cross-encoder models for document reranking
- Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Provides high-quality relevance scoring between query and documents

#### HybridReranker
- Combines multiple ranking strategies:
  - Cross-encoder scores (70% weight)
  - Keyword matching scores (20% weight)
  - Original retrieval scores (10% weight)
- Provides more robust ranking by considering multiple signals

#### Factory Function
- `create_reranker()` - Easy instantiation of different reranker types

### 2. Enhanced `hybrid_search.py`

The HybridSearch class now supports reranking:

```python
# Initialize with reranking enabled
search_engine = HybridSearch(
    collection_name=COLLECTION_NAME,
    enable_reranking=True,
    reranker_type="cross_encoder"  # or "hybrid"
)

# Search with reranking
results = search_engine.search(
    query="your query",
    k=5,
    rerank_top_k=10  # Rerank top 10, return top 5
)
```

## How It Works

1. **Initial Retrieval**: Hybrid search (vector + BM25) retrieves candidate documents
2. **Reranking**: Cross-encoder model scores query-document pairs for semantic relevance
3. **Final Selection**: Top-k most relevant documents are returned with updated scores

## Performance Impact

- **Latency**: Adds ~200-500ms for reranking 10 documents
- **Quality**: Significantly improves relevance, especially for complex queries
- **Memory**: Cross-encoder model requires ~100MB additional RAM

## Configuration Options

### HybridSearch Parameters

- `enable_reranking`: Enable/disable reranking (default: True)
- `reranker_type`: "cross_encoder" or "hybrid" (default: "cross_encoder")
- `rerank_top_k`: Number of documents to rerank (default: k*2)

### Reranker Models

You can customize the cross-encoder model:

```python
reranker = DocumentReranker(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")
```

Popular models:
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, good quality)
- `cross-encoder/ms-marco-MiniLM-L-12-v2` (slower, better quality)
- `cross-encoder/ms-marco-TinyBERT-L-2-v2` (fastest, lower quality)

## Test Results

All tests passed successfully:

```
✅ DocumentReranker: PASSED
✅ HybridReranker: PASSED  
✅ HybridSearch with Reranking: PASSED
```

Example reranking improvement:
- Query: "Python machine learning"
- Before: Generic Python docs ranked high
- After: ML-specific Python content ranked highest (score: 7.37 vs -0.58)

## Integration with RAG System

The reranker is automatically integrated into your existing RAG pipeline:

1. **RAGAssistant** uses HybridSearch for document retrieval
2. **HybridSearch** applies reranking when enabled
3. **Results** are returned with improved relevance scores
4. **No changes needed** to existing query processing logic

## Monitoring and Debugging

Reranking activities are logged:

```
INFO - Применяем реранжирование к 8 документам
INFO - Reranking 8 documents with cross-encoder  
INFO - Reranked 5 documents
INFO - Реранжирование завершено, возвращаем 5 документов
```

Document scores are preserved in metadata:
- `rerank_score`: Cross-encoder score
- `hybrid_rerank_score`: Combined hybrid score

## Best Practices

1. **Rerank Count**: Use `rerank_top_k = k * 2` for good balance of quality vs speed
2. **Model Selection**: Start with default model, upgrade if needed
3. **Fallback**: System gracefully falls back to original ranking if reranker fails
4. **Monitoring**: Watch logs for reranking performance and errors

## Troubleshooting

### Common Issues

1. **Model Download**: First run downloads the cross-encoder model (~100MB)
2. **Memory**: Ensure sufficient RAM for model loading
3. **Network**: Model download requires internet connection

### Disabling Reranking

If needed, disable reranking:

```python
search_engine = HybridSearch(enable_reranking=False)
```

## Future Enhancements

Potential improvements:
- Custom domain-specific reranker training
- Caching of reranked results
- A/B testing framework for ranking strategies
- Integration with user feedback for ranking improvement
