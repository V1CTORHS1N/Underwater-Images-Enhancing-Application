# Underwater Images Enhancing Application
This project aims to enhance underwater images using a Generative Adversarial Network (GAN) model, accompanied by a graphical user interface (GUI) for an improved user experience. 

The foundation of the project draws upon the FUnIE-GAN and WGAN-GP, with the model being trained on the EUVP dataset.

## Project Structure
```
.
├── README.md            # This file
├── main.py              # Main Application
├── dataset
│   └── EUVP             # Part of the EUVP Dataset
│       ├── GTr          # Ground Truth Images
│       └── Inp          # Input Images
│── icon
│   └── processing.gif   # Processing Icon
├── model
│   └── generator.pth    # Generator Model
├── net
│   └── net.py           # Network Architecture
├── images               # Sample Images
└── requirements.txt     # Required Packages
```
## How to Run
0. Clone the repository
   ```bash
    git clone https://github.com/V1CTORHS1N/Underwater-Images-Enhancing-Application.git
   ```
1. [Optional] Create and activate a virtual environment using [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
    ```bash
    conda create -n <env_name> python=3.10 && conda activate <env_name>
    ```
2. Install the required packages with following commands
    ```bash
    pip install -r /path/to/requirements.txt
    ```
3. Run the application
    ```bash
    python /path/to/main.py
    ```
4. Enjoy your hacking!

## Demo
### Enhance an Image
![Enhance an Image](./images/enhance(single).gif)

### Enhance Images in Batch
![Enhance Images in Batch](./images/enhance(batch).gif)

## References
- FUnIE-GAN: https://irvlab.cs.umn.edu/projects/funie-gan
- WGAN-GP: https://arxiv.org/abs/1704.00028
- EUVP Dataset: https://irvlab.cs.umn.edu/resources/euvp-dataset