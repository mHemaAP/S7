import torch

import matplotlib.pyplot as plt

##### Plot the train data #####
def display_train_data(train_data):

  print('[Train]')
  print(' - Numpy Shape:', train_data.cpu().numpy().shape)
  print(' - Tensor Shape:', train_data.size())
  print(' - min:', torch.min(train_data))
  print(' - max:', torch.max(train_data))
  print(' - mean:', torch.mean(train_data))
  print(' - std:', torch.std(train_data))
  print(' - var:', torch.var(train_data))

##### Show one image given index of the image #####
def show_image_by_index(images, index):  
  plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

##### Show multiple images #####
def display_multiple_images(images, num_of_images):
    
    figure = plt.figure()
  
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    
##### Plot model statistics #####        
def display_model_stats(train_loss, train_accuracy, test_loss, test_accuracy):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_loss)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_accuracy)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_loss)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_accuracy)
  axs[1, 1].set_title("Test Accuracy")


##### Plot incorrect test predictions #####  
def plot_test_incorrect_predictions(incorrect_pred):

    fig = plt.figure()
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.tight_layout()
        plt.imshow(incorrect_pred["images"][i].cpu().squeeze(0), cmap="gray")
        plt.title(
            repr(incorrect_pred["predicted_vals"][i])
            + " vs "
            + repr(incorrect_pred["ground_truths"][i])
        )
        plt.xticks([])
        plt.yticks([])
