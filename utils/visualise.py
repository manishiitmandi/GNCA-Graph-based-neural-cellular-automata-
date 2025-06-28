import matplotlib.pyplot as plt

def show_segmentation(img, pred_mask, gt_mask):
    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1)
    plt.imshow(img[0,0].cpu(), cmap='gray')
    plt.title('Input')
    plt.subplot(1,3,2)
    plt.imshow(pred_mask.detach().cpu(), cmap='gray')
    plt.title('Prediction')
    plt.subplot(1,3,3)
    plt.imshow(gt_mask.cpu(), cmap='gray')
    plt.title('Ground Truth')
    plt.show()
