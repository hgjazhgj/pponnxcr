import setuptools

with open("readme.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.readlines()

packages = setuptools.find_packages() + ['pponnxcr.model']

setuptools.setup(
    name="pponnxcr",
    version="2.0",
    author="hgjazhgj",
    author_email="hgjazhgj.jp@gmail.com",
    license='GNU AGPLv3',
    description="OCR based on onnxruntime with PaddleOCR models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hgjazhgj/pponnxcr",
    packages=packages,
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
