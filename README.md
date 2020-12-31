# Blank Language Models

This repository contains the code for our EMNLP 2020 paper:  
[**Blank Language Models**](https://arxiv.org/abs/2002.03079)  
*Tianxiao Shen&ast;, Victor Quach&ast;, Regina Barzilay, and Tommi Jaakkola (&ast;: Equal contribution)*

<br>

Given partially specified text with one or more blanks, BLM will fill in the blanks with a variable number of tokens consistent with the context, making it ideal for text editing and rewriting.

> Input:  They also have \___ which \___ .  
> Output: They also have <ins>ice cream</ins> which <ins>is really good</ins> .

<br>

<p align="center"><img width=900 src="img/model.png"></p>


## Demo

We have an online demo built using [streamlit](https://www.streamlit.io/), available [here](http://128.52.131.173:8501)

Or try locally by running:

```
streamlit run app.py
```


## Dependencies

Our code is based on the [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) framework.

It has been tested in PyTorch 1.6.0, PyTorch Lightning 1.0.7


## Download Data

Download the processed Yelp and Yahoo datasets by running:
```
bash download_data.sh
```


## Training

To train a BLM on Yelp negative sentences:
```
python train.py --train data/yelp/train.0 --valid data/yelp/valid.0 --root_dir checkpoints/yelp/neg/blm/ \
--vocab_size 10000 --max_len 20 --model_type blm --share_emb_prj_weight
```

Yelp positive sentences:
```
python train.py --train data/yelp/train.1 --valid data/yelp/valid.1 --root_dir checkpoints/yelp/pos/blm/ \
--vocab_size 10000 --max_len 20 --model_type blm --share_emb_prj_weight
```

Yahoo documents:
```
python train.py --train data/yahoo/train.txt --valid data/yahoo/valid.txt --root_dir checkpoints/yahoo/blm/ \
--vocab_size 20000 --max_len 205 --model_type blm --share_emb_prj_weight
```

Run `python train.py -h` to see all training options.

You can use Tensorboard to monitor the training progress.


## Testing

After training, we can evaluate the model's perplexity by Monte Carlo estimate, and use the model to generate text from scratch or fill in the blanks in the input.

For all of the following, replace `epoch\=???.ckpt` with the checkpoint saved in training.

- The following command evaluates for Yelp negative sentences:

```
python test.py --checkpoint checkpoints/yelp/neg/blm/lightning_logs/version_0/checkpoints/epoch\=???.ckpt \
--eval data/yelp/test.0 --n_mc 10
```

- The following command samples from the model trained on Yelp negative sentences:

```
python test.py --checkpoint checkpoints/yelp/neg/blm/lightning_logs/version_0/checkpoints/epoch\=???.ckpt \
--sample 1000 --decode sample --output sample.txt
```

- The following command uses the model trained on Yelp negative sentences to fill in blanked positive sentences to achieve sentiment transfer:

```
python test.py --checkpoint checkpoints/yelp/neg/blm/lightning_logs/version_0/checkpoints/epoch\=???.ckpt \
--fill data/yelp/blank/test.1.blank --output test.1.tsf
```

To output the whole generation trajectory, turn on the `--write_mid` option.

The output file will be stored in `outputs/` within the checkpoint directory.


## Acknowledgements

We use the Transformer implementation from https://github.com/jadore801120/attention-is-all-you-need-pytorch


## Citation

If you use our work, please cite:

```bibtex
@inproceedings{shen2020blank,
    title = "{Blank Language Models}",
    author = "Shen, Tianxiao  and
      Quach, Victor  and
      Barzilay, Regina  and
      Jaakkola, Tommi",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics"
}
```
