import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="runai",
    version="0.1.1",
    author="Run:AI",
    author_email="pypi@run.ai",
    description="Run:AI Python library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/run-ai/runai",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
    ],
)
