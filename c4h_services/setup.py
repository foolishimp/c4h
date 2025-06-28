"""
Setup configuration for c4h_services package.
Path: c4h_services/setup.py
"""

from setuptools import setup, find_packages

setup(
    name="c4h_services",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "c4h_agents",  # Depend on the agents package
        "prefect",
        "rich",
        "PyYAML",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0"
    ],
    python_requires=">=3.11",
)