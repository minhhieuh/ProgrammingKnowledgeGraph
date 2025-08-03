The experiments for the Context-Augmented Code Generation using Programming Knowledge Graphs (PKG) framework are conducted through a comprehensive setup that includes selecting specific code generation models, utilizing established benchmarks, implementing various retrieval methods, and meticulously generating the PKG itself from diverse datasets.

Here's a detailed breakdown of how the experiments are conducted:

### 1. Code Generation Models and Evaluation

The framework's performance is evaluated using a variety of Code Large Language Models (CLLMs) and a general-purpose Large Language Model (LLM).

*   **Models Used:** Experiments are conducted on four well-known CLLMs: CodeLlama-7B, CodeLlama-13B, StarCoder2-7B, and DeepSeek-Coder-7B. Additionally, Llama3.1-8B, a general-purpose LLM known for its strong performance in code generation, is also tested. All experiments are conducted using a single A100 GPU.
*   **Evaluation Metric:** The accuracy of the generated code is measured using the `pass@1` metric. This metric assesses whether the top-ranked generated solution correctly solves the problem.
*   **Decoding Approach:** A greedy decoding approach is adopted for the `pass@1` evaluation. This involves generating a single solution with a temperature setting of `t=0` and a token limit of 512 (maximum new tokens).

### 2. Benchmarks

To assess the general Python programming skills, problem-solving, and reasoning capabilities of both CLLMs and LLMs, two widely established datasets are used:

*   **HumanEval Dataset:** This dataset is used to evaluate the models' abilities.
*   **MBPP Benchmark:** This benchmark is also employed to assess problem-solving and reasoning in Python programming.

### 3. Retrieval Methods

The study utilizes two main retrieval methods for comparison and augmentation:

*   **Dense Retrieval:** The Voyage-Code-2 model is chosen for dense retrieval, recognized as a top-performing dense retriever for code. Embeddings for this method are obtained through API calls to the model.
*   **Sparse Retrieval:** The BM25 algorithm is employed for sparse retrieval. It is implemented using the `rank_bm25` Python library and exhibited the strongest performance among sparse retrieval techniques.

### 4. PKG Generation and Data Sources

The Programming Knowledge Graph (PKG) is generated from both code-centric and text-centric datasets.

*   **Dataset Selection:**
    *   **Code-centric Data:** The PythonAlpaca dataset is used, which consists of approximately 143,000 Python question-answer pairs. After preprocessing, 115,000 Python functions are extracted from this dataset.
    *   **Text-centric Data:** The Tutorials dataset, containing 76,600 programming tutorial text, is utilized.
*   **Block Extraction:**
    *   For code-centric data, an Abstract Syntax Tree (AST) parser called `FunctionAnalyzer` is used to extract function blocks from the dataset's output sections.
    *   For text-centric data, a JSON extraction language model, such as Gemma-2, is employed to generate JSON structures corresponding to the textual content.
*   **Graph Extraction:**
    *   **From Function Blocks:** Each code block (e.g., `if`, `for`, `with`, `try`) is represented as a node. The `FunctionAnalyzer` extracts the Context-Flow Graph (CFG) for each function, identifying code blocks as individual nodes. Three types of nodes are defined for each function: 'function name', 'function implementation', and 'extracted code blocks'. Structural edges capture relationships between these nodes, reflecting the code's hierarchical structure. Function-wise retrieval searches `Vimpl` nodes, while block-wise retrieval searches `Vblock` nodes.
    *   **From JSON Objects:** A JSON object `J` is represented as a graph `G_J` with nodes `V_J` and edges `E_J`. Each node `v` in `V_J` is a path-value pair, where the path is a concatenated string of keys from the root, serving as a unique identifier. Edges connect the current node to its embedded keys for JSON objects or to list items for arrays.
    *   The final PKG is constructed by aggregating these CFGs from code-centric datasets and Directed Acyclic Graphs (DAGs) from text-centric datasets.
*   **Encoding PKG:** The VoyageCode2 model is chosen to encode each node within the graph, enabling semantic search.
*   **Neo4j Graph Generation:** All extracted nodes, their embeddings, and relationships are imported into a Neo4j vector graph using the APOC plugin. The graphs are generated using Neo4j version 5.20.0. The PKG for PythonAlpaca comprised 425,058 nodes and 434,518 relations, while for the Tutorials dataset, it contained 288,583 path-value nodes and 287,936 relations.

### 5. Information Retrieval from PKG

To retrieve relevant information from the PKG for a given query, the following steps are performed:

*   **Query Embedding:** The user's query (`q`) is embedded into a d-dimensional space using the embedder model `E`.
*   **Semantic Vector Search:** A semantic vector search identifies the node (`v_best`) in the PKG most similar to the query by computing the cosine similarity between their embeddings.
*   **Retrieval Approaches:** Three distinct approaches are used:
    *   **Block-wise Retrieval ('Block-PKG'):** Operates on code blocks (`v_block`), focusing on granular context.
    *   **Function-wise Retrieval ('Func-PKG'):** Operates on function implementation nodes (`v_impl`), returning the entire function as context.
    *   **Path-value Retrieval ('JSON-PKG'):** Operates on path-value nodes from JSON objects.
*   **Branch Pruning:** After identifying the most similar node (`n_best`), irrelevant branches are removed to ensure only useful information is passed to the generative model. This involves modeling `n_best` as a DAG and selecting a pruned version that maximizes cosine similarity with the query embedding.
*   **Query Augmentation:** The original query is augmented with the content of the `n_pruned` graph, forming `q_augmented` for the generative model.

### 6. Solution Re-ranking

A re-ranking mechanism is implemented to address hallucination and select the best solution from multiple candidates generated by different approaches (e.g., Block-PKG, Func-PKG, No RAG).

*   **Step 1: AST Analysis (Syntactic Filtering):** An AST analysis function (`A`) filters out syntactically erroneous solutions from the initial candidates (`C`), yielding a subset of syntactically valid samples (`C_A`).
*   **Step 2: Runtime Execution (Runtime Error Filtering):** The syntactically valid candidates (`C_A`) are executed, and a runtime execution function (`R`) eliminates solutions with runtime issues, resulting in `C_R` (candidates that execute without errors).
*   **Step 3: Semantic Similarity Check:** The embeddings of the remaining candidates (`C_R`) are compared with the query embedding using cosine similarity. The solution with the highest similarity score (`c*`) is returned as the final selected solution. This step ensures prioritization of solutions generated without reliance on potentially erroneous RAG-based content, mitigating the impact of hallucinations.

### 7. Prompt Templates

Specific prompt templates are used for each model to guide the generation process, incorporating augmented data when PKG is utilized.

*   **CodeLlama-7B Prompt:** The prompt instructs the model to act as a Python programmer, solve the given problem, and integrate helpful augmented code if useful, otherwise ignore it. The solution must be enclosed in `[PYTHON]` and `[/PYTHON]` tags and be executable.
*   **StarCoder2-7B Prompt:** This prompt uses an "Instruction" and "Response" format, asking the model to solve the problem as a Python programmer and integrate helpful code from the `augmented_data` section.
*   **DeepSeek-Coder-7B Prompt:** Similar to CodeLlama, this prompt frames the request within `[INST]` tags, instructing the model to solve the problem as a Python programmer and integrate helpful code if provided.

### 8. Cost Trade-off Considerations

The time and storage usage for creating RAG data sources are also considered as part of the experimental setup. PKG generation on the PythonAlpaca dataset takes 301 minutes and 12,530 MB of storage, which is higher than VoyageAI (241 minutes, 8,440 MB) or BM25 (44 minutes, 315 MB). However, this trade-off is justified by PKG's average 9.4% higher accuracy. Neo4j's semantic vector indexing allows efficient graph updates with logarithmic complexity, and queries typically take about 3 seconds.