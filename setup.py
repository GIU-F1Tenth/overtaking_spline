import os
from glob import glob

from setuptools import find_packages, setup

package_name = "overtaking_spline"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name), ["LICENSE"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Karim Shousha",
    maintainer_email="karim.shousha.ks@gmail.com",
    description="Frenet-frame quintic-spline overtaking planner for F1TENTH.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "overtaking_spline_node = overtaking_spline.overtaking_spline_node:main",
        ],
    },
)
