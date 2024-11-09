# ERVQA: A Benchmark Dataset for Assessing Large Vision-Language Models in Hospital Environments

**[Read the ERVQA Paper Here](https://arxiv.org/pdf/2410.06420)**

## Table of Contents
- [Code Structure](#code-structure)
- [Installation](#installation)
- [Data Access](#data-access)


## Code Structure

This repository includes the following core modules:

- **`src/classifier.py`**: Implements the BLIP-2 model for error classification, tailored for healthcare-related image assessments.
- **`src/eval.py`**: Contains modules for calculating key metrics, including Entailment Score and CLIPScore Confidence, essential for evaluating VLM predictions' reliability.

## Installation

To set up the required environment, please install the dependencies as follows:

```bash
pip install torch==2.3.0 transformers==4.41.2 Pillow==10.3.0 sentence-transformers==3.0.1
```

## Data Access

We are currently figuring out our data release policy as the ERVQA dataset may contain sensitive information. If you wish to access the data urgently, please send a mail to [sourjyadipray@gmail.com](mailto:sourjyadipray@gmail.com). 

