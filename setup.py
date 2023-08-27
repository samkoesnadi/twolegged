from setuptools import setup, find_packages

setup(
    name="twolegged",
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'gym',
        'pybullet',
        'numpy',
        'matplotlib',
        'torch',
        'stable_baselines3 @ git+https://github.com/DLR-RM/stable-baselines3',
        'sb3_contrib @ git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib',
        'imitation @ git+https://github.com/samuelkoes/imitation'
    ]
)
