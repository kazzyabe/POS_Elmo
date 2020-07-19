import matplotlib.pyplot as plt

history = {'loss': [0.26695565565312207, 0.07284288032919414, 0.055323470083704175, 0.04405510949725208, 0.03239121828382173], 'accuracy': [0.9302996, 0.97826445, 0.98311985, 0.9860796, 0.9893957], 'val_loss': [0.2235705812390034, 0.17708400464974916, 0.1887159634094972, 0.1702739206644205, 0.17804221321757024], 'val_accuracy': [0.9408654, 0.94600964, 0.9436058, 0.9485577, 0.94721156]}
# summarize history for accuracy
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()