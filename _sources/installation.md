# Installation
We recommend installing the `pip`-based package through a virtual environment such as `conda` to separate from the usage of other environments or packages.

1. [Install Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) if you don't have it already.

2. Open a terminal and create a new virtual environment named `kim` with Python 3.11:
   ```bash
   conda create --name kim python=3.11
   ```

3. Activate the newly created virtual environment:
   ```bash
   conda activate kim
   ```

4. Install the package using pip within the activated virtual environment:
   ```bash
   pip install kim-jax
   ```

5. (Optional) Download the git repo to get the example jupyter notebooks:
    ```bash
    git clone https://github.com/PeishiJiang/KIM.git
    ```