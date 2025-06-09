You can install the required packages via:
```bash
pip install -r requirements.txt
```



## run CAT
```
python main.py --dataset imdb --num_clients 5 --algorithm cat --num_rounds 10 --local_epochs 1 --batch_size 4 --epsilon 0.05
```

## run CAT2

```
python main.py --dataset imdb --num_clients 5 --algorithm cat2 --confidence_threshold 0.9 --batch_threshold 0.9 --num_rounds 10 --local_epochs 1 --batch_size 4 --epsilon 0.05
```
