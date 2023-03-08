import os
import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


class GoToML:
    def __init__(self) -> None:
        (
            self.DEVICE,
            self.tokenizer,
            self.train_data,
            self.test_data,
        ) = self._initialize_data()

        (
            self.vocab,
            self.text_pipeline,
            self.label_pipeline,
        ) = self._initialize_vocabulary()

        self.model = self._initialize_model()

        pass

    def _initialize_data(self):
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = get_tokenizer("basic_english")

        train_data = pd.read_csv(f"{os.getcwd()}/goto_ml/training_data.csv")
        test_data = pd.read_csv(f"{os.getcwd()}/goto_ml/test_data.csv")

        return DEVICE, tokenizer, train_data, test_data

    def _initialize_vocabulary(self):
        train_data, tokenizer = self.train_data, self.tokenizer

        train_data_iter = train_data.itertuples()

        vocab = build_vocab_from_iterator(
            self._yield_tokens(train_data_iter, tokenizer),
            specials=["<unk>"],
        )
        vocab.set_default_index(vocab["<unk>"])

        text_pipeline = lambda x: vocab(tokenizer(x))
        label_pipeline = lambda x: int(x) - 1

        return vocab, text_pipeline, label_pipeline

    def _initialize_model(self):
        self.class_map = {
            1: "/clusters",
            2: "/security/database/users",
            3: "/security/network/accessList",
            4: "/dataFederation",
            5: "/clusters/atlasSearch",
            6: "/security/advanced",
            7: "/dataLake",
            8: "/billing/overview",
        }

        train_data, vocab = self.train_data, self.vocab
        train_iter = train_data.itertuples()

        label_set = set([url for (_, url, text) in train_iter])

        num_class = len(label_set)
        vocab_size = len(vocab)
        emsize = 64
        model = TextClassificationModel(vocab_size, emsize, num_class).to(
            self.DEVICE
        )

        return model

    def train_and_evaluate(self):
        model, train_data, test_data = (
            self.model,
            self.train_data,
            self.test_data,
        )

        EPOCHS = 1000  # epoch
        LR = 5  # learning rate
        BATCH_SIZE = 64  # batch size for training

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
        total_accu = None

        train_iter = train_data.itertuples()
        test_iter = test_data.itertuples()

        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)

        num_train = int(len(train_dataset) * 0.95)
        split_train_, split_valid_ = random_split(
            train_dataset, [num_train, len(train_dataset) - num_train]
        )

        train_dataloader = DataLoader(
            split_train_,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=self._collate_batch,
        )
        valid_dataloader = DataLoader(
            split_valid_,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=self._collate_batch,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=self._collate_batch,
        )

        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()
            self._train(train_dataloader, optimizer, criterion, epoch)
            accu_val = self._evaluate(valid_dataloader, criterion)
            if total_accu is not None and total_accu > accu_val:
                scheduler.step()
            else:
                total_accu = accu_val
            print("-" * 59)
            print(
                "| end of epoch {:3d} | time: {:5.2f}s | "
                "valid accuracy {:8.3f} ".format(
                    epoch, time.time() - epoch_start_time, accu_val
                )
            )
            print("-" * 59)

        print("Checking the results of test dataset.")
        accu_test = self._evaluate(test_dataloader, criterion)
        print("test accuracy {:8.3f}".format(accu_test))

    def predict(self, text):
        with torch.no_grad():
            text = torch.tensor(self.text_pipeline(text))
            output = self.model(text, torch.tensor([0]))

            class_index = output.argmax(1).item() + 1

            print(
                "We think you should visit the "
                f"{self.class_map[class_index]} page."
            )

            return

    def _train(self, dataloader, optimizer, criterion, epoch):
        self.model.train()
        total_acc, total_count = 0, 0
        log_interval = 500

        for idx, (label, text, offsets) in enumerate(dataloader):
            optimizer.zero_grad()
            predicted_label = self.model(text, offsets)
            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches "
                    "| accuracy {:8.3f}".format(
                        epoch, idx, len(dataloader), total_acc / total_count
                    )
                )
                total_acc, total_count = 0, 0

    def _evaluate(self, dataloader, criterion):
        self.model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predicted_label = self.model(text, offsets)
                _ = criterion(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc / total_count

    def _collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for _, _label, _text in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(
                self.text_pipeline(_text), dtype=torch.int64
            )
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)

        return (
            label_list.to(self.DEVICE),
            text_list.to(self.DEVICE),
            offsets.to(self.DEVICE),
        )

    def _yield_tokens(self, data_iter, tokenizer):
        for data in data_iter:
            text = data[2]
            yield tokenizer(text)


if __name__ == "__main__":
    goto_ml = GoToML()

    goto_ml.train_and_evaluate()
    model = goto_ml.model
