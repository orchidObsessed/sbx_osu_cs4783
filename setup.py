from setuptools import setup

setup(
        name="ass2",
        version="0.1",
        url="https://github.com/orchidObsessed/sbx-osu-cs4783",
        author="William \"Waddles\" Waddell",
        license="None",
        packages=["helpers", "neural", ],
        include_package_data=True,
        install_requires=["gym", "numpy", "opencv-python", "pillow"]
)
