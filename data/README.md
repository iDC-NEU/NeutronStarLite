You can use `generate_nts_dataset.py` to generate the files needed for training.

```bash
# install package
pip install -r requirements.txt 

# Usage
# --dataset DATASET     Dataset name (cora, citeseer, pubmed, reddit)
# --self-loop SELF_LOOP insert self-loop (default=True)

generate_nts_dataset.py --dataset=reddit
```
