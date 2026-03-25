import matplotlib.pyplot as plt

epochs = range(1, 13)
train_losses = [0.9916, 0.8875, 0.7561, 0.7609, 0.6654, 0.7028, 0.6842, 0.6298, 0.5662, 0.5447, 0.5014, 0.5411]
val_losses = [0.7716, 0.8683, 0.9423, 1.0008, 1.1295, 1.1559, 1.1851, 1.2761, 1.0734, 1.3247, 1.3938, 1.3325]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
plt.axvline(x=1, color='g', linestyle='--', label='Best Model (Epoch 1)')
plt.title('Learning Curve: Overfitting Analysis')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()