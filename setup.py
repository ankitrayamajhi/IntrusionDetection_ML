from setuptools import setup, find_packages

setup(
    name='intrusion_detection_system',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib',
        'matplotlib',
        'seaborn',
    ],
    entry_points={
        'console_scripts': [
            'preprocess_data=scripts.preprocess_data:main',
            'train_scaler=scripts.train_scaler:main',
            'train_gbm=scripts.train_gbm:main',
            'train_svm=scripts.train_svm:main',
            'train_rf=scripts.train_rf:main',
            'train_knn=scripts.train_knn:main',
            'evaluate_model=scripts.evaluate_model:main',
        ],
    },
)
