from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent
README = HERE.joinpath("README.md").read_text()
REQUIREMENTS = HERE.joinpath("requirements", "requirements.in").read_text().split()

setup(
    # Metadata
    name="x-mlps",
    author="Miller Wilt",
    author_email="miller@pyriteai.com",
    use_scm_version=True,
    description="Configurable MLPs built on JAX and Haiku",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/PyriteAI/x-mlps",
    keywords=["artificial intelligence", "machine learning", "jax", "haiku", "multilayer perceptron", "mlp"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
    # Options
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=REQUIREMENTS,
    python_requires=">=3.8.0",
)
