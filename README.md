This repo implements the [pytorch text classifier tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html) to classify search strings against urls on MongoDB Cloud.

To get started, initialize a [poetry](https://python-poetry.org/docs/) venv and install dependencies with `poetry shell && poetry install`.  Then run `python goto_ml/goto_ml.py` in the environment.  This will utilize the pre-trained model weights from `goto_ml.pt`.

If you'd like to see the model re-trained (non-deterministic) on your machine, or train it based on an expanded dataset, replace the training data `.csv` files and initialize the `GoToML` class with `train=True`.  This will re-train the model and save the state in `goto_ml.pt`.

You can predict a url based on your input using the class's `predict` function.  Output from this function is as follows:

```
2023-03-08 16:29:22 [info     ] Sorry, we don't know which page you should visit. confidence=tensor(0.4651) minimum_confidence=tensor(0.5000) prompt=charlie

2023-03-08 16:29:22 [info     ] We think you should visit the /security/network/accessList page. confidence=tensor(0.9956) minimum_confidence=tensor(0.5000) prompt=ip address
```

This basic exploratory project was done during one of the [twice-yearly hackathons](https://www.mongodb.com/blog/post/skunkworks-2022-week-building-mongodb-engineers) here at MongoDB.