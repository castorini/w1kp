import setuptools

setuptools.setup(
    name='w1kp',
    version=eval(open('w1kp/_version.py').read().strip().split('=')[1]),
    author='Raphael Tang',
    license='MIT',
    url='https://github.com/castorini/w1kp',
    author_email='r33tang@uwaterloo.ca',
    description='Words worth a thousand pictures',
    install_requires=open('requirements.txt').read().strip().splitlines(),
    packages=setuptools.find_packages(),
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'w1kp_generate_images=w1kp.run.generate_images:main',
            'w1kp_trainer=w1kp.run.trainer:main',
        ]
    }
)
