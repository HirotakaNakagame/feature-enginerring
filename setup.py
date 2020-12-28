from setuptools import setup

setup(
        name="feature-engineering",
        version="0.0.1",
        author="Hirotaka Nakagame",
        author_email="hirotaka.nakagame@gmail.com",
        packages=["fe"],
        package_dir={"fe":"fe"},
        url="https://github.com/HirotakaNakagame/feature-enginerring/",
        license="MIT",
        install_requires=[  "sklearn >= 0.24"]
)
