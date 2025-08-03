"""
Core PKG implementation modules.

This package contains the main components for:
- Function analysis and code block extraction
- Function enhancement with FIM objectives
- Programming Knowledge Graph generation
- Solution re-ranking mechanisms
- Code generation utilities
"""

# Only import lightweight modules by default
from .function_analyzer import FunctionAnalyzer

# Heavy dependency modules can be imported explicitly when needed:
# from .function_enhancer import FunctionEnhancer
# from .reranker import rerank_one_solution, cosine_similarity 