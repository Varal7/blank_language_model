# blank_language_model

```
CUDA_VISIBLE_DEVICES=3 python train.py \
--train /data/rsg/nlp/tianxiao/blank_language_model/data/yahoo/train.txt --valid /data/rsg/nlp/tianxiao/blank_language_model/data/yahoo/valid.txt \
--vocab_size 20000 --max_len 200 \
--share_emb_prj_weight \
--weight_decay 1e-6 --dropout 0.3 \
--lr 1e-4 --train_steps 500000 \
--max_tok 8000 --eval_max_tok 20000 --checkpoint_every 5000 \
--save_dir /data/rsg/nlp/tianxiao/blank_language_model/checkpoints/yahoo/insertion_share_wd1e-6_drop0.3_lr1e-4_train500k/
```

```
CUDA_VISIBLE_DEVICES=0 python train.py \
--train /data/rsg/nlp/tianxiao/blank_language_model/data/penn/train.txt --valid /data/rsg/nlp/tianxiao/blank_language_model/data/penn/valid.txt \
--vocab_size 10000 --max_len 100 \
--share_emb_prj_weight \
--weight_decay 1e-5 --dropout 0.5 \
--lr 5e-4 --lr_schedule linear_decay --train_steps 100000 --warmup_steps 50000 \
--max_tok 10000 --eval_max_tok 40000 --checkpoint_every 2000 \
--save_dir /data/rsg/nlp/tianxiao/blank_language_model/checkpoints/penn/insertion_nocat_share_wd1e-5_drop0.5_lr5e-4_lineardecay_train100k_warmup50k_tok10000/
```

```
CUDA_VISIBLE_DEVICES=0 python test.py \
--checkpoint /data/rsg/nlp/tianxiao/blank_language_model/checkpoints/wikitext-2/winsertion_share_wd1e-5_drop0.3_lr2e-4_lineardecay_train300k_warmup50k_tok4000x2 \
--eval /data/rsg/nlp/tianxiao/blank_language_model/data/wikitext-2/test.txt.part1 \
--n_mc 1000
```

```
CUDA_VISIBLE_DEVICES=1 python test.py \
--checkpoint /data/rsg/nlp/tianxiao/blank_language_model/checkpoints/wikitext-2/winsertion_share_wd1e-5_drop0.3_lr2e-4_lineardecay_train300k_warmup50k_tok4000x2 \
--eval /data/rsg/nlp/tianxiao/blank_language_model/data/wikitext-2/valid.txt \
--n_mc 10
```

```
CUDA_VISIBLE_DEVICES=2 python test.py \
--checkpoint /data/rsg/nlp/tianxiao/blank_language_model/checkpoints/wikitext-2/winsertion_share_wd1e-5_drop0.3_lr2e-4_lineardecay_train300k_warmup50k_tok4000x2 \
--eval /data/rsg/nlp/tianxiao/blank_language_model/data/wikitext-2/valid.txt \
--n_mc 100
```


```
CUDA_VISIBLE_DEVICES=1 python test.py \
--checkpoint /data/rsg/nlp/tianxiao/blank_language_model/checkpoints/wikitext-2/winsertion_share_wd1e-5_drop0.3_lr2e-4_lineardecay_train300k_warmup50k_tok4000x2 \
--eval /data/rsg/nlp/tianxiao/blank_language_model/data/wikitext-2/test.txt \
--n_mc 1000
```



```
CUDA_VISIBLE_DEVICES=0 python test.py \
--checkpoint /data/rsg/nlp/tianxiao/blank_language_model/checkpoints/penn/insertion_nocat_share_wd1e-5_drop0.5_lr5e-4_lineardecay_train100k_warmup50k_tok10000 \
--eval /data/rsg/nlp/tianxiao/blank_language_model/data/penn/test.txt \
--n_mc 1
```

```
CUDA_VISIBLE_DEVICES=2 python test.py \
--checkpoint /data/rsg/nlp/tianxiao/blank_language_model/checkpoints/penn/insertion_nocat_share_wd1e-5_drop0.5_lr5e-4_lineardecay_train100k_warmup50k_tok10000 \
--eval /data/rsg/nlp/tianxiao/blank_language_model/data/penn/test.txt \
--n_mc 10
```


```
CUDA_VISIBLE_DEVICES=0 python test.py \
--checkpoint /data/rsg/nlp/tianxiao/blank_language_model/checkpoints/penn/insertion_nocat_share_wd1e-5_drop0.5_lr5e-4_lineardecay_train100k_warmup50k_tok10000 \
--eval /data/rsg/nlp/tianxiao/blank_language_model/data/penn/test.txt \
--n_mc 1000
```

```
CUDA_VISIBLE_DEVICES=1 python test.py \
--checkpoint /data/rsg/nlp/tianxiao/blank_language_model/checkpoints/penn/insertion_nocat_share_wd1e-5_drop0.5_lr5e-4_lineardecay_train100k_warmup50k_tok10000 \
--eval /data/rsg/nlp/tianxiao/blank_language_model/data/penn/test.txt \
--n_mc 100
```

```
CUDA_VISIBLE_DEVICES=1 python train.py \
--train /data/rsg/nlp/tianxiao/blank_language_model/data/wikitext-2/train.txt --valid /data/rsg/nlp/tianxiao/blank_language_model/data/wikitext-2/valid.txt \
--vocab_size 40000 --max_len 705 \
--share_emb_prj_weight \
--weight_decay 1e-5 --dropout 0.3 \
--lr 2e-4 --lr_schedule linear_decay --train_steps 600000 --warmup_steps 100000 \
--max_tok 4000 --eval_max_tok 8000 --checkpoint_every 5000 \
--save_dir /data/rsg/nlp/tianxiao/blank_language_model/checkpoints/wikitext-2/insertion_share_wd1e-5_drop0.3_lr2e-4_lineardecay_train600k_warmup100k_tok2000/
```

```
CUDA_VISIBLE_DEVICES=1 python train.py \
--train /data/rsg/nlp/tianxiao/blank_language_model/data/wikitext-2/train.txt --valid /data/rsg/nlp/tianxiao/blank_language_model/data/wikitext-2/valid.txt \
--vocab_size 40000 --max_len 705 \
--share_emb_prj_weight \
--weight_decay 1e-5 --dropout 0.3 \
--lr 2e-4 --lr_schedule linear_decay --train_steps 600000 --warmup_steps 100000 \
--accum_grad 2 --max_tok 4000 --eval_max_tok 10000 --checkpoint_every 5000 \
--save_dir /data/rsg/nlp/tianxiao/blank_language_model/checkpoints/wikitext-2/winsertion_share_wd1e-5_drop0.3_lr2e-4_lineardecay_train300k_warmup50k_tok4000x2
```


```
CUDA_VISIBLE_DEVICES=1 python fill.py \
--data /data/rsg/nlp/tianxiao/blank_language_model/data/yahoo/infill/test.maskratio0.2.blank \
--output infill/test.maskratio0.2.fill \
--checkpoint /data/rsg/nlp/tianxiao/blank_language_model/checkpoints/yahoo/insertion_share_wd1e-6_drop0.3_lr1e-4_train500k/
```

```
CUDA_VISIBLE_DEVICES=0 python train.py \
--train /data/rsg/nlp/tianxiao/blank_language_model/data/wikitext-103/train.bpe --valid /data/rsg/nlp/tianxiao/blank_language_model/data/wikitext-103/valid.bpe \
--vocab_size 42000 --max_len 1500 \
--accum_grad 8 \
--share_emb_prj_weight \
--weight_decay 1e-5 --dropout 0.3 \
--lr 1e-4 --train_steps 1000000 \
--max_tok 2500 --eval_max_tok 10000 --checkpoint_every 5000 \
--save_dir /data/rsg/nlp/tianxiao/blank_language_model/checkpoints/wikitext-103/bpe_insertion_layer6_nocat_share_wd1e-5_drop0.3_lr1e-4_train1M_tok2500x4
```

```
CUDA_VISIBLE_DEVICES=3 python test.py \
--checkpoint /data/rsg/nlp/tianxiao/blank_language_model/checkpoints/wikitext-103/bpe_insertion_layer6_nocat_share_wd1e-5_drop0.3_lr1e-4_train1M_tok2500x4 \
--eval /data/rsg/nlp/tianxiao/blank_language_model/data/wikitext-103/sortest_test.bpe.part0 \
--n_mc 1000 --output test.bpe.part0.n_nc.1000
```

# Infill reborn

```
CUDA_VISIBLE_DEVICES=0 python fill.py \
--data /data/rsg/nlp/tianxiao/blank_language_model/data/yahoo/infill/test.maskratio0.1.blank \
--output infill/test.maskratio0.1.fill \
--checkpoint /data/rsg/nlp/tianxiao/blank_language_model/checkpoints/yahoo/insertion_share_wd1e-6_drop0.3_lr1e-4_train500k/
```

```
CUDA_VISIBLE_DEVICES=1 python fill.py \
--data /data/rsg/nlp/tianxiao/blank_language_model/data/yahoo/infill/test.maskratio0.2.blank \
--output infill/test.maskratio0.2.fill \
--checkpoint /data/rsg/nlp/tianxiao/blank_language_model/checkpoints/yahoo/insertion_share_wd1e-6_drop0.3_lr1e-4_train500k/
```

```
CUDA_VISIBLE_DEVICES=2 python fill.py \
--data /data/rsg/nlp/tianxiao/blank_language_model/data/yahoo/infill/test.maskratio0.3.blank \
--output infill/test.maskratio0.3.fill \
--checkpoint /data/rsg/nlp/tianxiao/blank_language_model/checkpoints/yahoo/insertion_share_wd1e-6_drop0.3_lr1e-4_train500k/
```
