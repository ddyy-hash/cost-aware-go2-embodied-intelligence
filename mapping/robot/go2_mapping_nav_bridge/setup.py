from glob import glob
from setuptools import find_packages, setup

package_name = "go2_mapping_nav_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Unitree Developer",
    maintainer_email="unitree@example.com",
    description="Mission-to-navigation bridge and ROS 2 Nav2 execution adapter for Go2 mapping.",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "navigation_executor = go2_mapping_nav_bridge.navigation_executor:main",
        ],
    },
)
