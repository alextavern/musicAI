import matplotlib.pyplot as plt
def plot_histories(histories, title, epochs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for experiment in histories:
        train_losses = histories[experiment][0]
        val_losses = histories[experiment][1]
        train_accuracies = histories[experiment][2]
        val_accuracies = histories[experiment][3]

        x_axis = range(1, epochs + 1)
        ax1.plot(x_axis, train_losses, label='train loss:' + str(experiment))
        ax1.plot(x_axis, val_losses, label='validation loss:' + str(experiment))
        ax1.set_xlabel("Epoch #")
        ax1.set_ylabel("Loss (a.u.)")
        # ax1.legend()
        ax2.plot(x_axis, train_accuracies, label='train accuracy:' + str(experiment))
        ax2.plot(x_axis, val_accuracies, label='validation accuracy:' + str(experiment))
        ax2.set_xlabel("Epoch #")
        ax2.set_ylabel("Accuracy (a.u.)")
        # ax2.legend()
        fig.suptitle("ESC10: MLP w mels (base) - " + title)

    plt.savefig("ESC10_MLP-mels-base_" + title + ".png")


def plot_history(histories, experiment, epochs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    train_losses = histories[experiment][0]
    val_losses = histories[experiment][1]
    train_accuracies = histories[experiment][2]
    val_accuracies = histories[experiment][3]

    x_axis = range(1, epochs + 1)
    ax1.plot(x_axis, train_losses, label='train loss:' + str(experiment))
    ax1.plot(x_axis, val_losses, label='validation loss:' + str(experiment))
    ax1.set_xlabel("Epoch #")
    ax1.set_ylabel("Loss (a.u.)")
    # ax1.legend()
    ax2.plot(x_axis, train_accuracies, label='train accuracy:' + str(experiment))
    ax2.plot(x_axis, val_accuracies, label='validation accuracy:' + str(experiment))
    ax2.set_xlabel("Epoch #")
    ax2.set_ylabel("Accuracy (a.u.)")
    ax2.legend()
    fig.suptitle("ESC10: MLP w mels (base) - " + str(experiment))

    plt.savefig("ESC10_MLP-mels-base_" + str(experiment) + ".png")
    files.download("ESC10_MLP-mels-base_" + str(experiment) + ".png")