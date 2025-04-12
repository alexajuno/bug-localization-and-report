from collections import namedtuple
from pathlib import Path

# Dataset root directory
_DATASET_ROOT = Path('./data')
_SOURCE_FILES = _DATASET_ROOT / 'source-files'
_BUG_REPORTS = _DATASET_ROOT / 'bug-reports'

Dataset = namedtuple('Dataset', ['name', 'src', 'bug_repo', 'repo_url'])

# Source codes and bug repositories

aspectj = Dataset(
    'aspectj',
    _SOURCE_FILES / 'org.aspectj-bug433351',
    _BUG_REPORTS / 'AspectJ.txt',
    "https://github.com/eclipse/org.aspectj/tree/bug433351.git"
)

eclipse = Dataset(
    'eclipse',
    _SOURCE_FILES / 'eclipse.platform.ui-johna-402445',
    _BUG_REPORTS / 'Eclipse_Platform_UI.txt',
    "https://github.com/eclipse/eclipse.platform.ui.git"
)

swt = Dataset(
    'swt',
    _SOURCE_FILES / 'eclipse.platform.swt-xulrunner-31',
    _BUG_REPORTS / 'SWT.txt',
    "https://github.com/eclipse/eclipse.platform.swt.git"
)

tomcat = Dataset(
    'tomcat',
    _SOURCE_FILES / 'tomcat-7.0.51',
    _BUG_REPORTS / 'Tomcat.txt',
    "https://github.com/apache/tomcat.git"
)

birt = Dataset(
    'birt',
    _SOURCE_FILES / 'birt-20140211-1400',
    _BUG_REPORTS / 'Birt.txt',
    "https://github.com/eclipse/birt"
)

### Current dataset in use. (change this name to change the dataset)
DATASET = swt

if __name__ == '__main__':
    print(DATASET.name, DATASET.src, DATASET.bug_repo)
