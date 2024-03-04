# DP-CRE

Source code of the Decouple Processed Continual Relation Extraction  (DP-CRE) [Decouple Processed to Balance Continual Relation Extraction and Preserve Memory Structure].

>Continuous Relation Extraction (CRE) aims to incrementally learn relation knowledge from a non-stationary stream of data. One significant challenge in this domain is catastrophic forgetting, where the introduction of new relational tasks can overshadow previously learned information. Unlike traditional replay-based training paradigms that uniformly prioritize all data, we decouple the process of prior information preservation and new knowledge acquisition. In this paper, we introduce the Decouple Processed CRE (DP-CRE) framework. This approach examines alterations in the embedding space as new relation classes emerge, distinctly managing the preservation and acquisition of knowledge. Extensive experiments show that DP-CRE significantly outperforms other CRE
baselines across two datasets.

# Environment
- Python 3.7.16
- PyTorch: 1.13.1(cuda version 11.7).  
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```
To install PyTorch, you could follow the official guidance of [PyTorch](https://pytorch.org/). 

Then, other dependencies could be installed by running:
```
pip install -r requirements.txt
```

Pre-trained BERT weights:

* Download *bert-base-uncased* into the *root* directory [[google drive]](https://drive.google.com/drive/folders/1BGNdXrxy6W_sWaI9DasykTj36sMOoOGK).

## Dataset

We use `FewRel` and `TACRED` datasets in our experiments.
- FewRel: `data/data_with_marker.json`
- TACRED: `data/data_with_marker_tacred.json`

The splited datasets and task orders is conducted in `sample.py`.

## Usage

To reproduce the results of main experiment:
```
python main.py --task_name FewRel
python main.py --task_name TACRED
```