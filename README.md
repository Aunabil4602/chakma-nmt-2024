# chakma-nmt-2024


# Dataset
The dataset is the most crucial aspect of this work. The complete dataset, named "<a href=""><i>chakma-nmt-complete-dataset</a>,"</i> includes <b>parallel</b>, <b>monolingual</b>, and <b>benchmark</b> sets for Chakma to Bangla or English translations, and vice versa.
<br>
Details about the <b>parallel</b> set (exact corresponding translations):
<ul>
    <li>Total Bangla-Chakma-English parallel sentences/segments: 8647</li>
    <li>Only Bangla-Chakma parallel sentences/segments: 6374</li>
    <li>Total Bangla-Chakma parallel sentences/segments: (8647+6374) = 15021</li>
</ul>

To train the Bangla-Chakma and Chakma-Bangla translations, we divided the complete Bangla-Chakma parallel dataset into two distinct subsets:
<ul>
    <li><a href=""><i>chakma-nmt-base-parallel-train-set</i></a>: This set serves as the primary training data for Chakma-Bangla translations and vice versa, comprising 80% of the total parallel dataset.</li>
    <li><a href=""><i>chakma-nmt-base-parallel-dev-set</i></a>: This set is used for evaluation during training, consisting of 20% of the parallel dataset to assess Chakma-Bangla translations and vice versa.</li>
</ul>


Details of the <b>monolingual</b> set (No corresponding translations):
<ul>
    <li>Total Bangla monolingual sentences/segments: 150000</li>
    <li>English monolingual sentences/segments: 150000</li>
    <li>Total Chakma monolingual sentences/segments: 42783</li>
</ul>

Details about the <b>benchmark</b> dataset:<br>
Benchmark set consists of 600 samples of parallel (corresponding translations) Chakma-Bangla sentences/segments.

# Codes
First install the libraries with specific versions as mentioned in the requirements.txt file. To install the libraries open your terminal and the following script:<br>
`pip install requirements.txt`

This repository includes three files:
<ul>
  <li><i>ck_nmt_final_rnn+trans.py</i>: To train and test RNN and Transformer models (both from scratch) for translating between Chakma and Bangla.</li>
  <li><i>cn_nmt_banglat5_trainer.py</i>: To train and test with the pretrained <a href="https://huggingface.co/csebuetnlp/banglat5">BanglaT5</a> model for Chakma and Bangla translation.</li>
  <li><i>multilingual_cn_nmt_banglat5_trainer</i>: To train and test with the pretrained <a href="https://huggingface.co/csebuetnlp/banglat5">BanglaT5</a>, incorporating Chakma, Bangla, and English languages.</li>
</ul>

Each of the files has a class named <i>"CONFIG"</i>. It has the necessary hyper-parameters and other fields for training. Change the values accordingly and run the files.