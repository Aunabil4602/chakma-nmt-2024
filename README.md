# ChakmaNMT

## Paper

This repository contains code and resources for **[ChakmaNMT: Machine Translation for a Low-Resource and Endangered Language via Transliteration](https://arxiv.org/abs/2410.10219)**, accepted at **WMT 2026**.

The work introduces the first and, to our knowledge, largest documented Chakma--Bangla machine translation resource. It studies from-scratch neural and statistical MT, pretrained models, multilingual training, back-translation, and transliteration-based transfer for Chakma (`ccp`) and Bangla (`bn`).

## Dataset

The complete dataset is available on Hugging Face:

**[amlan107/chakma-nmt-complete-dataset](https://huggingface.co/datasets/amlan107/chakma-nmt-complete-dataset)**

It contains three splits:

| Split | Size | Description |
|---|---:|---|
| Parallel | 15,021 | Bangla--Chakma translation pairs, comprising 9,548 sentence pairs and 5,473 dictionary word pairs. Of the sentence pairs, 8,647 also include aligned English. |
| Monolingual | Up to 150,000 per language | 150,000 Bangla, 150,000 English, and 42,783 Chakma samples. |
| Benchmark | 600 | Chakma--Bangla--English evaluation rows corresponding to 500 unique source sentences. |

For the bilingual experiments, the 15,021 parallel pairs were divided into:

- **Training:** [12,016 samples](https://huggingface.co/datasets/amlan107/chakma-nmt-base-parallel-train-set)
- **Development:** [3,005 samples](https://huggingface.co/datasets/amlan107/chakma-nmt-base-parallel-dev-set)

The multilingual experiments additionally use 10,000 Bangla--English translation pairs from [*Not Low-Resource Anymore: Aligner Ensembling, Batch Filtering, and New Datasets for Bengali-English Machine Translation*](https://aclanthology.org/2020.emnlp-main.207/).

## Code

Install the required packages with:

```bash
pip install -r requirements.txt
```

The `train/` directory contains:

- `ck_nmt_final_rnn_trans.py`: trains and evaluates the from-scratch RNN and Transformer models.
- `cn_nmt_banglat5_trainer.py`: trains and evaluates BanglaT5 for Chakma--Bangla translation.
- `multilingual_cn_nmt_banglat5_trainer.py`: trains and evaluates multilingual BanglaT5 with Chakma, Bangla, and English.

Each training script provides a `CONFIG` class for paths and hyperparameters and reports performance on the benchmark set.

The normalization and Chakma--Bangla transliteration code is maintained separately in **[chakma-nmt-normalizer](https://github.com/Aunabil4602/chakma-nmt-normalizer)**.

## Citation

If you use the dataset or code, please kindly cite our paper:

```bibtex
@misc{chakma2026chakmanmtmachinetranslationlowresource,
      title={ChakmaNMT: Machine Translation for a Low-Resource and Endangered Language via Transliteration}, 
      author={Aunabil Chakma and Aditya Chakma and Masum Hasan and Soham Khisa and Chumui Tripura and Rifat Shahriyar},
      year={2026},
      eprint={2410.10219},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.10219}, 
}
```
