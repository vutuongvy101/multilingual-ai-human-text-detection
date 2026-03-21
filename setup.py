#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multilingual-ai-detection",
    version="0.1.0",
    author="Tuong Vy Vu",
    author_email="your.email@example.com",
    description="Multilingual AI-Human Text Detection using ML and Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multilingual-ai-human-text-detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
        ],
        "data": [
            "beautifulsoup4>=4.12.0",
            "lxml>=4.9.0",
            "praw>=7.7.0",
            "requests>=2.31.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-statistical=multilingual_ai_detection.scripts.train_statistical:main",
            "train-transformer=multilingual_ai_detection.scripts.train_transformer:main",
            "infer=multilingual_ai_detection.scripts.infer:main",
            "serve-api=multilingual_ai_detection.scripts.serve_api:main",
            "web-demo=multilingual_ai_detection.scripts.web_demo:main",
        ],
    },
)