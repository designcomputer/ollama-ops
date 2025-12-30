from setuptools import setup

setup(
    name='ollama-ops',
    version='0.1.0a1',
    description='A robust utility for managing and auto-optimizing local Ollama models.',
    author='Dana Williams',
    author_email='dana@designcomputer.com',  # Placeholder based on your context
    py_modules=['ollama_ops'],  # Assumes you renamed the file to ollama_ops.py
    install_requires=[
        'ollama>=0.1.6',
    ],
    entry_points={
        'console_scripts': [
            'ollama-ops=ollama_ops:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
