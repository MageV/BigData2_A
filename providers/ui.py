import matplotlib.pyplot as plt


def do_plot_train_trees(model,name):
    logs = model.make_inspector().training_logs()
    fig=plt.figure(figsize=(12, 4))
    fig.canvas.manager.set_window_title(name)
    plt.subplot(1, 2, 1)
    plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy (out-of-bag)")
    plt.subplot(1, 2, 2)
    plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Logloss (out-of-bag)")
    plt.show()


def do_plot_history_seq(history,name):
    # Plotting Training and Validation Loss
    fig=plt.figure()
    fig.canvas.manager.set_window_title(name)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
