from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tumorfs",
    version="0.1.0",
    author="Your Name",
    author_email="your@email.com",
    description="Multi-algorithm feature selection for tumor RNA-seq classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/<your-username>/TumorFS",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "r": ["rpy2>=3.4"],
        "dev": ["pytest>=7.0", "pytest-cov", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "tumorfs=tumorfs.cli:main",
        ],
    },
)
