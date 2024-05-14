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


def do_plot_history_seq(history, name, metric=None):
    # Plotting Training and Validation Loss
    if metric is None:
        metric = ['accuracy']
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
    for item in metric:
        plt.plot(history.history[item], label=f'Training {item} ')
        plt.plot(history.history[f'val_{item}'], label=f'Validation {item}')
        plt.ylabel(f'{item}')
    plt.title(f'Training and Validation {metric}')
    plt.xlabel('Epochs')
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
