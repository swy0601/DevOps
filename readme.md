# LDA topic modeling

## Contents

1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Usage

### Please see code comments for more details

### Quick start

#### Pre-process

1. Download stack overflow dataset
   from [dataset](https://drive.google.com/drive/folders/1p-TJ3J0u4hZmyZ9ZD6io-YBAJ6XzyihD?usp=sharing).
      
2. Run the following script.

```bash
    python pre-process.py
```

#### Train

1. (Optional) Run the following script to get perplexity. 

```bash
    python perplexity.py
```

2. Run the following script to get LDA model.

```bash
    python train_LDA_model.py
```

## Detail Package Version

```yaml
    numpy==1.21.6
    pandas==1.3.5
    
    pickle==1.1.1
    gensim==3.8.3
```
