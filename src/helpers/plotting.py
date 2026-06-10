import matplotlib.pyplot as plt


def plot_multiple_clients(samples, client_indices, step, writer):
    """
    Plot the generated load curves of several clients (full year + first week)
    and log the resulting figures to TensorBoard.
    """
    fig_full, axes_full = plt.subplots(len(client_indices), 1, figsize=(14, 3.5 * len(client_indices)))
    for i, client_idx in enumerate(client_indices):
        x_shape = samples[client_idx]
        x_recon = x_shape
        x_recon_curve = x_recon.flatten().detach().cpu().numpy()

        ax = axes_full[i] if len(client_indices) > 1 else axes_full
        ax.plot(x_recon_curve, label=f"Client {client_idx}", alpha=0.8)
        ax.set_title(f"Full-year Consumption - Client {client_idx} - Step {step}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    writer.add_figure("ClientGeneration/FullYear_MultipleClients", fig_full, global_step=step)
    plt.close(fig_full)

    fig_week, axes_week = plt.subplots(len(client_indices), 1, figsize=(14, 3.5 * len(client_indices)))
    for i, client_idx in enumerate(client_indices):
        x_shape = samples[client_idx]
        x_recon = x_shape
        x_recon_curve = x_recon.flatten().detach().cpu().numpy()
        ax = axes_week[i] if len(client_indices) > 1 else axes_week
        ax.plot(x_recon_curve[:336], label=f"Client {client_idx}", alpha=0.8)
        ax.set_title(f"Week 1 - Client {client_idx} - Step {step}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    writer.add_figure("ClientGeneration/Week1_MultipleClients", fig_week, global_step=step)
    plt.close(fig_week)