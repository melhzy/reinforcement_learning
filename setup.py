"""
Setup configuration for the RL Framework.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rl-alzheimers-framework",
    version="0.1.0",
    author="RL Framework Contributors",
    description="Reinforcement Learning Framework for LLM + Alzheimer's Disease Research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/melhzy/reinforcement_learning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core framework has no external dependencies
    ],
    extras_require={
        "llm": ["openai>=1.0.0", "anthropic>=0.5.0"],
        "viz": ["matplotlib>=3.5.0", "seaborn>=0.12.0"],
        "data": ["numpy>=1.21.0", "pandas>=1.3.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
    include_package_data=True,
    package_data={
        "rl_framework": ["configs/*.json"],
    },
)
