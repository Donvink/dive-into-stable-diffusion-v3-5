from setuptools import setup
import os

def read_requirements():
    with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="dive-into-stable-diffusion",
    version="0.1",
    install_requires=read_requirements(),
    # 其他参数...
)