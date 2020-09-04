from pathlib import Path

import setuptools

CURRENT_DIR = Path(__file__).parent

setuptools.setup(
    name="support_resistance_line",
    version="0.0.7",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    description=(
        "A well-tuned algorithm to generate & draw support/resistance line on time series. "
        "根据时间序列自动生成支撑线压力线"
    ),
    long_description=(CURRENT_DIR / "README.md").read_text(encoding="utf8"),
    long_description_content_type="text/markdown",
    author="WestXu",
    author_email="xu-lai-xi@qq.com",
    url="https://github.com/WestXu/support_resistance_line",
    install_requires=["matplotlib", "numpy", "pandas", "sklearn", "lazy_object_proxy"],
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
