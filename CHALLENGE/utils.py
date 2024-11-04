import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from matplotlib import pyplot as plt


# Consente di visualizzare un confronto tra l’immagine di input, la DepthMap reale e quella predetta. 
# Usa una mappa di colore magma per evidenziare le variazioni di profondità, e la pausa di 20 secondi 
# permette di esaminare ogni immagine.
def visualize_img(img_tensor, depth_tensor, pred_tensor, suffix):
    img = img_tensor.numpy().transpose(1, 2, 0)  # Convert to numpy array and transpose
    gt = depth_tensor.numpy().transpose(1, 2, 0)  # Convert to numpy array and transpose
    pred = pred_tensor.numpy().transpose(1, 2, 0)  # Convert to numpy array and transpose
    # Create a figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img)
    axs[0].set_title('Input')
    axs[1].imshow(gt, cmap='magma')
    axs[1].set_title(f'True')
    axs[2].imshow(pred, cmap='magma')
    axs[2].set_title(f'Predicted')
    fig.suptitle(suffix)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(20)
    plt.close()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

# (Structural Similarity Index): Calcola la somiglianza strutturale tra la DepthMap predetta e quella ground truth.
# La SSIM è particolarmente utile per mantenere la qualità strutturale e i dettagli durante la predizione, ed è 
# efficace come metrica complementare all'RMSE o al MAE.
def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
