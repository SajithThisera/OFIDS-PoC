import torch
import torch.optim as optim
from transformers import VideoMAEForVideoClassification


def train_videomae_model(train_features, train_labels, learning_rate=1e-4, epochs=10):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base").to("cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    train_features = torch.tensor(train_features, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_features).logits
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    return model
