import tensorflow as tf
import matplotlib.pyplot as plt
import pickle 

with open('history.pkl', 'rb') as f:
    history = pickle.load(f)

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
#plot_graphs(history, "accuracy")
plot_graphs(history, "loss")