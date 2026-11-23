from glob import glob
import os
from setuptools import find_packages, setup

package_name = "wp3_live_perception"

setup(
    name=package_name,
    version="2.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.json")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Shadi / ENGR7001 Project 7",
    maintainer_email="80166316+erfrifrung@users.noreply.github.com",
    description="Display-only WP3 Jenga perception overlay wrapping wp3_live.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "wp3_live_node = wp3_live_perception.live_node:main",
        ],
    },
)
