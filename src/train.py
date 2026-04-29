import nltk
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW


nltk.download("punkt")


def identify_metaphor_sentence(text, metaphor_word):
    for sentence in nltk.sent_tokenize(text):
        if metaphor_word.lower() in sentence.lower():
            return sentence
    return None


def main():
    df = pd.read_csv("train.csv")

    df = df.dropna(subset=["text", "label_boolean"])

    metaphor_map = {
        0: "road",
        1: "candle",
        2: "light",
        3: "spice",
        4: "ride",
        5: "train",
        6: "boat"
    }

    df["metaphorID"] = df["metaphorID"].replace(metaphor_map)

    df["text"] = df.apply(
        lambda row: identify_metaphor_sentence(row["text"], row["metaphorID"]),
        axis=1
    )

    df = df.rename(columns={"metaphorID": "metaphor_word"})
    df = df.dropna(subset=["text", "label_boolean"])

    
    
    #df = df.sample(200, random_state=42)

    train_data, test_data = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label_boolean"]
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    train_tokens = tokenizer(
        list(train_data["text"]),
        list(train_data["metaphor_word"]),
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    test_tokens = tokenizer(
        list(test_data["text"]),
        list(test_data["metaphor_word"]),
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    train_dataset = TensorDataset(
        train_tokens["input_ids"],
        train_tokens["attention_mask"],
        torch.tensor(train_data["label_boolean"].astype(int).values)
    )

    test_dataset = TensorDataset(
        test_tokens["input_ids"],
        test_tokens["attention_mask"],
        torch.tensor(test_data["label_boolean"].astype(int).values)
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 10 == 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    print("\nModel Performance")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nClassification Report")
    print(classification_report(true_labels, predictions, zero_division=0))

    model.save_pretrained("saved_model")
    tokenizer.save_pretrained("saved_model")

    print("\nModel saved to saved_model/")


if __name__ == "__main__":
    main()