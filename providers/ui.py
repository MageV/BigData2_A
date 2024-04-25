import matplotlib.pyplot as plt
import seaborn as sns


def do_plot_train_trees(model, name):
    logs = model.make_inspector().training_logs()
    fig = plt.figure(figsize=(12, 4))
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


def do_plot_history_seq(history, name, metric='accuracy'):
    # Plotting Training and Validation Loss
    fig = plt.figure(figsize=(12, 4))
    fig.canvas.manager.set_window_title(name)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history[metric], label=f'Training {metric} ')
    plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
    plt.title(f'Training and Validation {metric}')
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric}')
    plt.legend()
    plt.show()


def do_plot_conf_mx(con_mat_df, name):
    figure = plt.figure(figsize=(8, 8))
    figure.canvas.manager.set_window_title(name)
    sns.heatmap(con_mat_df, annot=True)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
