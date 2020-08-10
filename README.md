# blank_language_model

```

python train.py \
  --project_name varal7/blm \
  --train /data/rsg/nlp/quach/blank_project/blank_language_model/data/phi-ml/train.txt \
  --valid /data/rsg/nlp/quach/blank_project/blank_language_model/data/phi-ml/valid.txt \
  --name ancient-insT \
  --model_type inst \
  --add_eos \
  --cat_sent \
  --max_len 1024 \
  --max_tok 8192 \
  --checkpoint_every 0 \
  --vocab_size 200 \
  --n_mc 0 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --share_emb_prj_weight \
  --accum_grad 8 \
  --dropout 0.3 \
  --lr_schedule fixed \
  --train_steps 6000000 \
  --root_dir /data/scratch/quach/serialize/phi-ml/inst/6000000

```

```

python train.py \
  --project_name varal7/blm \
  --train /data/rsg/nlp/quach/blank_project/blank_language_model/data/phi-ml/train.txt \
  --valid /data/rsg/nlp/quach/blank_project/blank_language_model/data/phi-ml/valid.txt \
  --name ancient-lblm \
  --model_type lblm \
  --add_eos \
  --cat_sent \
  --max_len 1024 \
  --max_tok 8192 \
  --checkpoint_every 0 \
  --vocab_size 200 \
  --n_mc 0 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --share_emb_prj_weight \
  --accum_grad 8 \
  --dropout 0.3 \
  --lr_schedule fixed \
  --train_steps 6000000 \
  --root_dir /data/scratch/quach/serialize/phi-ml/lblm/6000000

```


```

CUDA_VISIBLE_DEVICES=0 python train.py \
  --project_name varal7/blm \
  --train /data/rsg/nlp/tianxiao/blank_language_model/data/wikitext-103/train.txt.small \
  --valid /data/rsg/nlp/tianxiao/blank_language_model/data/wikitext-103/valid.txt.small \
  --name insT-debug \
  --model_type inst \
  --add_eos \
  --cat_sent \
  --max_len 256 \
  --max_tok 16384 \
  --checkpoint_every 0 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --share_emb_prj_weight \
  --accum_grad 8 \
  --dropout 0.3 \
  --lr_schedule fixed \
  --warmup_steps 30000 \
  --train_steps 2000000 \
  --fp16 --fp16_opt_level O2 \
  --root_dir /data/scratch/quach/serialize/blank_project/debug-inst

```

```

CUDA_VISIBLE_DEVICES=1 python train.py \
  --project_name varal7/blm \
  --train /data/rsg/nlp/tianxiao/blank_language_model/data/wikitext-103/train.bpe \
  --valid /data/rsg/nlp/tianxiao/blank_language_model/data/wikitext-103/valid.bpe \
  --name blm-debug \
  --model_type blm \
  --add_eos \
  --cat_sent \
  --max_len 256 \
  --max_tok 16384 \
  --checkpoint_every 0 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --share_emb_prj_weight \
  --accum_grad 8 \
  --dropout 0.3 \
  --lr_schedule fixed \
  --warmup_steps 30000 \
  --train_steps 2000000 \
  --fp16 --fp16_opt_level O2 \
  --root_dir /data/scratch/quach/serialize/blank_project/debug-blm

```


```

CUDA_VISIBLE_DEVICES=2 python train.py \
  --project_name varal7/blm \
  --train /data/rsg/nlp/tianxiao/blank_language_model/data/wikitext-103/train.txt.small \
  --valid /data/rsg/nlp/tianxiao/blank_language_model/data/wikitext-103/valid.txt.small \
  --name lblm-debug \
  --model_type lblm \
  --add_eos \
  --cat_sent \
  --max_len 256 \
  --max_tok 16384 \
  --checkpoint_every 0 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --share_emb_prj_weight \
  --accum_grad 8 \
  --dropout 0.3 \
  --lr_schedule fixed \
  --warmup_steps 30000 \
  --train_steps 2000000 \
  --fp16 --fp16_opt_level O2 \
  --root_dir /data/scratch/quach/serialize/blank_project/debug-lblm
```

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

CUDA_VISIBLE_DEVICES=0 python fill.py \
--data /data/rsg/nlp/quach/blank_project/blank_language_model/data/phi-ml/valid.blank_x \
--constrained_length_single_blank \
--output valid.x.infull \
--checkpoint /data/scratch/quach/serialize/phi-ml/inst/6000000/ancient-insT/version_BLM-182/

CUDA_VISIBLE_DEVICES=1 python fill.py \
--data /data/rsg/nlp/quach/blank_project/blank_language_model/data/phi-ml/test.blank_x \
--constrained_length_single_blank \
--output test.x.infull \
--checkpoint /data/scratch/quach/serialize/phi-ml/inst/6000000/ancient-insT/version_BLM-182/

python auxiliary/extract_infill.py --blank /data/rsg/nlp/quach/blank_project/blank_language_model/data/phi-ml/valid.blank_x --full /data/scratch/quach/serialize/phi-ml/inst/6000000/ancient-insT/version_BLM-182/valid.x.infull > /data/scratch/quach/serialize/phi-ml/inst/6000000/ancient-insT/version_BLM-182/valid.x.infill

python auxiliary/extract_infill.py --blank /data/rsg/nlp/quach/blank_project/blank_language_model/data/phi-ml/valid.blank_x_small --full /data/scratch/quach/serialize/phi-ml/inst/6000000/ancient-insT/version_BLM-182/valid.blank_x_small.infull > /data/scratch/quach/serialize/phi-ml/inst/6000000/ancient-insT/version_BLM-182/valid.blank_x_small.infill

python auxiliary/error_rate.py --pred /data/scratch/quach/serialize/phi-ml/inst/6000000/ancient-insT/version_BLM-182/valid.blank_x_small.infill --gold /data/rsg/nlp/quach/blank_project/blank_language_model/data/phi-ml/valid.y
