# How does Grammatical Gender Affect Noun Representations in Gender-Marking Languages?

This project includes the experiments described in the [paper](https://www.aclweb.org/anthology/K19-1043/): 

**"How does Grammatical Gender Affect Noun Representations in Gender-Marking Languages?"**  
Hila Gonen, Yova Kementchedjhieva and Yoav Goldberg, CoNLL 2019 (best paper).

## Prerequisites

* [word2vecf](https://github.com/BIU-NLP/word2vecf)
* For german: [DEMorphy](https://github.com/DuyguA/DEMorphy)

## Debiased embeddings - ready to use

To use the non-debiased and debiased embeddings in German and in Italian, download the files from this [folder](https://drive.google.com/drive/folders/1PuiNC-XTsQadMWKyYRxXzCxOKoOWdAEJ?usp=sharing) (under `embeddings`, 8 files):

* de: nondebiased, German
* it: nondebiased, Italian
* de_lemma_basic_all: debiased, German
* it_lemma_to_fem: debiased, Italian

These files are preprossesed for fast loading. To load them, use the script `source/save_embeds.py`:

```sh
vocab = {}  
wv = {}  
w2i = {}  
load_and_normalize('en', path, vocab, wv, w2i)
```

## Training debiased embeddings

If you want to train debiased embeddings from scratch, here are the steps to take:

* Download the corpora from this [folder](https://drive.google.com/drive/folders/1PuiNC-XTsQadMWKyYRxXzCxOKoOWdAEJ?usp=sharing) (under `data`) into the `data` folder (or use your own)

* Use the script `create_pairs_word2vecf.py`.
	
	This will create a dictionary that converts every word in the corpus to its new form. Using this dictionary, the script will create files of pairs for word2vecf.
	
	Usage for German:

	```sh
	python create_pairs_word2vecf.py --lang de --lemmatize basic --input ../data/de_corpus_tokenized --output ../data/word2vecf_pairs/de_lemma_basic_all
	```

	Usage for Italian:
	
	```sh
	python create_pairs_word2vecf.py --lang it --lemmatize to_fem --input ../data/it_corpus_tokenized --output ../data/word2vecf_pairs/it_lemma_to_fem
	```

* Next, run word2vecf (scripts can be found [here](https://bitbucket.org/yoavgo/word2vecf/src/default/)):
	
	Usage example for German:
	```sh
	./count_and_filter -train ../data/word2vecf_pairs/de_lemma_basic_all -cvocab ../data/word2vecf_pairs/de_lemma_basic_all_cv -wvocab ../data/word2vecf_pairs/de_lemma_basic_all_wv -min-count 100
	
	./word2vecf -train ../data/word2vecf_pairs/de_lemma_basic_all -wvocab ../data/word2vecf_pairs/de_lemma_basic_all_wv -cvocab  ../data/word2vecf_pairs/de_lemma_basic_all_cv -output ../data/embeddings/word2vecf/de_lemma_basic_all -dumpcv ../data/embeddings/word2vecf/de_lemma_basic_all_ctx -size 300 -negative 15 -threads 40 -iters 5
	```
	
	
## Simlex-999 inanimate pairs

Can be found under the `data` folder.

## Cite

If you find this project useful, please cite the paper:
```
@inproceedings{grammatical_gonen,
    title = "How Does Grammatical Gender Affect Noun Representations in Gender-Marking Languages?",
    author = "Gonen, Hila and Kementchedjhieva, Yova and Goldberg, Yoav",
    booktitle = "Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)",
    year = "2019",
    pages = "463--471",
}
```

## Contact

If you have any questions or suggestions, please contact [Hila Gonen](mailto:hilagnn@gmail.com).

## License

This project is licensed under Apache License - see the [LICENSE](LICENSE) file for details.


