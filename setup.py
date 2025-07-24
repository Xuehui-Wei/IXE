from setuptools import setup, find_packages

setup(
    name="IXE_package",  
    version="0.1.0",    
    author="Xuehui Wei",
    author_email="xwei47@asu.edu",
    description="This package provides tools for image processing and spectrum analysis of XES collected al LCLS.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Xuehui-Wei/IXE",
    packages=find_packages(include=['IXE', 'IXE.*']),  
    package_dir={'IXE': 'IXE'}, 
    package_data={'IXE': ['*.py']},
    entry_points={
        'console_scripts': [
            'xes-analyzer=IXE.xes_analyzer:run_gui'
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Python版本要求
    install_requires=[        # 依赖项
        "numpy>=1.0",
        "scipy>=1.0",
        "matplotlib>=3.0",
        "Pillow>=9.0",
        "scikit-image>=0.19",
    ],
)