from setuptools import setup, find_packages

setup(
    name="kimi_audio",
    packages=find_packages(include=["app*", "kimia_infer*"]),
)