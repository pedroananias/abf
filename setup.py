from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent

__version__ = ""
exec((this_directory / "src" / "abf" / "version.py").read_text(encoding="utf-8"))

# This is THE standard way to share packages in Python world
# see https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html
setup(
    name="abf",
    version=__version__,
    author="Pedro Ananias and Rogerio Negri",
    author_email="pedro.ananias@unesp.br",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    url="https://github.com/pedroananias/abf",
    license=(this_directory / "LICENSE").read_text(encoding="utf-8"),
    description="Anomalous Behaviour Forecast forecasts anomalies occurences based on "
                "images from Google Earth Engine API and machine learning.",
    long_description=(this_directory / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",  # Optional (see note above)
    python_requires=">=3.7.7,<3.10",
    install_requires=[
        "oauth2client==4.1.3",
        "earthengine-api==0.1.328",
        "matplotlib==3.6.1",
        "pandas==1.5.1",
        "numpy==1.23.4",
        "requests==2.28.1",
        "Pillow==9.2.0",
        "natsort==8.2.0",
        "geojson==2.5.0",
        "joblib==1.2.0",
        "scipy==1.9.3",
        "scikit-learn==1.1.2",
        "black==22.10.0",
        "click==8.1.3",
        "jupyterlab==3.4.8",
        "seaborn==0.12.0",
        "tensorflow==2.10.0",
        "black==22.10.0",
        "jupyterlab==3.4.8",
        "loguru==0.6.0"
    ],
    extras_require={},
    entry_points={
        "console_scripts": ["abf=abf.cli:forecast"],
    },
    zip_safe=False,
    include_package_data=True,
)
