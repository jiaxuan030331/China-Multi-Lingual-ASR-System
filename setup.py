from setuptools import setup, find_packages
import os

# 找到所有包
def find_all_packages():
    packages = []
    
    # 添加主包
    packages.extend(find_packages())
    
    # 添加kimi_deployment下的包
    kimi_base = 'kimi_deployment'
    if os.path.exists(kimi_base):
        for root, dirs, files in os.walk(kimi_base):
            if '__init__.py' in files:
                package_path = root.replace('/', '.').replace('\\', '.')
                packages.append(package_path)
    
    return packages

setup(
    name="asr",
    version="0.1.0",
    description="Integrated ASR System with Whisper and Kimi",
    packages=find_all_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "transformers",
        "librosa", 
        "loguru",
        "ctranslate2",
        "faster-whisper",
        "huggingface_hub",
    ],
) 