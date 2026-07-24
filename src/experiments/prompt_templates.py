#!/usr/bin/env python3
"""
Centralized Prompt Templates for PKG Experiments

This module contains all prompt templates used across the PKG experiment system.
Modify these templates to change prompts globally across all experiments.
"""

# System Prompt Template
SYSTEM_PROMPT_TEMPLATE = """You are an expert Python programmer. Your task is to solve programming problems by writing clean, executable Python code.

Requirements:
- Write executable Python code
- Include all necessary imports
- Ensure the solution is self-contained
- Write your solution between [PYTHON] and [/PYTHON] tags
- No explanation, no extra text, no comments outside of the codes

"""

# User Prompt Templates
USER_PROMPT_TEMPLATE_WITH_CONTEXT = """Solve the following problem:

{problem}

The following code might be helpful as reference:
{context}

If the helper code is useful, integrate its logic directly into your solution. Otherwise, ignore it.
"""

USER_PROMPT_TEMPLATE_NO_CONTEXT = """Solve the following problem:

{problem}
"""

# Legacy Full Prompt Templates (for backward compatibility)
FULL_PROMPT_TEMPLATE_WITH_CONTEXT = """You are an expert Python programmer. Solve the following problem:

{problem}

The following code might be helpful as reference:
{context}

If the helper code is useful, integrate its logic directly into your solution. Otherwise, ignore it.
"""

FULL_PROMPT_TEMPLATE_NO_CONTEXT = """You are an expert Python programmer. Solve the following problem:

{problem}
"""

# Alternative Templates (for experimentation)
ALTERNATIVE_SYSTEM_PROMPT = "You are a skilled software engineer specializing in Python development. Focus on writing efficient, readable, and well-documented code."

ALTERNATIVE_USER_PROMPT_NO_CONTEXT = """Please implement a solution for the following programming challenge:

{problem}
"""

# Template Configuration
class PromptConfig:
    """Configuration class for selecting prompt templates"""
    
    def __init__(self, 
                 system_template: str = SYSTEM_PROMPT_TEMPLATE,
                 user_template_with_context: str = USER_PROMPT_TEMPLATE_WITH_CONTEXT,
                 user_template_no_context: str = USER_PROMPT_TEMPLATE_NO_CONTEXT):
        self.system_template = system_template
        self.user_template_with_context = user_template_with_context
        self.user_template_no_context = user_template_no_context
    
    def get_system_prompt(self) -> str:
        """Get the system prompt template"""
        return self.system_template
    
    def get_user_prompt(self, problem: str, context: str = None) -> str:
        """Get the user prompt with or without context"""
        if context:
            return self.user_template_with_context.format(problem=problem, context=context)
        else:
            return self.user_template_no_context.format(problem=problem)
    
    def get_full_prompt(self, problem: str, context: str = None) -> str:
        """Get the complete prompt (system + user) as a single string"""
        if context:
            return FULL_PROMPT_TEMPLATE_WITH_CONTEXT.format(problem=problem, context=context)
        else:
            return FULL_PROMPT_TEMPLATE_NO_CONTEXT.format(problem=problem)

# Default configuration
DEFAULT_PROMPT_CONFIG = PromptConfig()

# Alternative configuration for experimentation
ALTERNATIVE_PROMPT_CONFIG = PromptConfig(
    system_template=ALTERNATIVE_SYSTEM_PROMPT,
    user_template_no_context=ALTERNATIVE_USER_PROMPT_NO_CONTEXT
) 