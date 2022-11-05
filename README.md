# EASumm


<!-------------------------------------------------------------------------------->

## Overview

Code and data accompanying the paper ["Graph-Enhanced Biomedical Abstractive Summarization via Factual Evidence Extraction"](todo), extended by ["Enhancing Biomedical Scientific Reviews Summarization with Graph-based Factual Evidence Extracted from Papers"](https://www.scitepress.org/PublicationsDetail.aspx?ID=/jornliCVuw=&t=1) (Best Studen Paper Award @ DATA22).

EASumm is the first abstractive summarization model augmenting source documents with explicit, structured medical evidence extracted from them, thereby concretizing a tandem text-graph architecture.

<p align="center">
  <img src="./figures/overview.png" title="EASumm architecture overview" alt="EASumm architecture overview">
</p>

## Install requirements
```
pip install -r requirements.txt
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-geometric
```

## Download extracted events with DeepEventMine
```
cd deep_event_mine
gdown 1x3oHfAKdtYfTEKuLPFTV_b2foA-VEMSx
```

## Train our model
```
python train_abstractor.py --wandb_log
```

## Decode
```
python decode_abstractor.py  --model_dir ckpts
```

## Evaluate 

Download ROUGE-1.5.5 and tell pyrouge the ROUGE path
```
gdown 1Df0FY4k-EGbvOlIBk2-Ih7J5N5ss-Ko4
tar -xvf ROUGE.tar.gz
rm ROUGE.tar.gz
pyrouge_set_rouge_path $(pwd)/ROUGE
```

```
python eval_full_model.py  --decode_dir ckpts 
```

## ✉ Contacts

* Giacomo Frisoni, [giacomo.frisoni[at]unibo.it](mailto:giacomo.frisoni@unibo.it)
* Paolo Italiani, [paolo.italiani[at]studio.unibo.it](mailto:paolo.italiani2@unibo.it)
* Gianluca Moro, [gianluca.moro[at]unibo.it](mailto:gianluca.moro@unibo.it)

If you have troubles, suggestions, or ideas, the [Discussion](https://github.com/disi-unibo-nlp/easumm/discussions) board might have some relevant information. If not, you can post your questions there 💬🗨.


<!-------------------------------------------------------------------------------->

## License

This project is released under the CC-BY-NC-SA 4.0 license (see `LICENSE`).

## Cite

If you use EASumm in your research, please cite:

      @inproceedings{DBLP:conf/data/FrisoniIBM22,
        author    = {Giacomo Frisoni and
                    Paolo Italiani and
                    Francesco Boschi and
                    Gianluca Moro},
        editor    = {Alfredo Cuzzocrea and
                    Oleg Gusikhin and
                    Wil M. P. van der Aalst and
                    Slimane Hammoudi},
        title     = {Enhancing Biomedical Scientific Reviews Summarization with Graph-based Factual Evidence Extracted from Papers},
        booktitle = {Proceedings of the 11th International Conference on Data Science,
                    Technology and Applications, {DATA} 2022, Lisbon, Portugal, July 11-13,
                    2022},
        pages     = {168--179},
        publisher = {{SCITEPRESS}},
        year      = {2022},
        url       = {https://doi.org/10.5220/0011354900003269},
        doi       = {10.5220/0011354900003269},
        timestamp = {Wed, 03 Aug 2022 15:53:22 +0200},
        biburl    = {https://dblp.org/rec/conf/data/FrisoniIBM22.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
      }
