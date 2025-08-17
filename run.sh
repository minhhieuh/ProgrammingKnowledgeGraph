# python -m src.experiments.experiment_runner --model-name claude-3-haiku-20240307 --model-type anthropic --benchmark humaneval --augmentation-types no_rag bm25 voyage_func voyage_block --verbose
python -m src.experiments.experiment_runner --model-name gpt-4o-mini --model-type openai --benchmark mbpp --augmentation-types no_rag bm25 voyage_func voyage_block --verbose
