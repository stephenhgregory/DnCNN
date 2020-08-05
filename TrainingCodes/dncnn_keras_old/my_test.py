# Run this script to test the model

import argparse
import os, time, datetime
# import PIL.Image as Image
import numpy as np
from keras.models import load_model, model_from_json
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
import tensorflow as tf

# Set Memory Growth to true to fix a small bug in Tensorflow

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    print(f'The following line threw an exception: tf.config.experimental.set_memory_growth(physical_devices[0], True)')
    pass


#############################################################


def parse_args():
    """
    Parses Command Line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data/CoregisteredImages', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['test'], type=list, help='name of test dataset')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    parser.add_argument('--model_dir', default=os.path.join('models', 'DnCNN_sigma25'), type=str,
                        help='directory of the model')
    parser.add_argument('--model_name', default='model_007.hdf5', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of results')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()


def to_tensor(image):
    """ Converts an input image to a tensor """

    if image.ndim == 2:
        print('The number image dimensions is 2!')
        return image[np.newaxis, ..., np.newaxis]
    elif image.ndim == 3:
        print('The number of image dimensions is 3!')
        return np.moveaxis(image, 2, 0)[..., np.newaxis]


def from_tensor(img):
    """ Converts an image tensor into an image """

    return np.squeeze(np.moveaxis(img[..., 0], 0, -1))


def log(*args, **kwargs):
    """ Generic logger function to print current date and time """

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path):
    """ Saves an image or file to a specific path """

    path = path if path.find('.') != -1 else path + '.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))


def show(x, title=None, cbar=False, figsize=None):
    """ Creates a matplotlib plot of an input image x """

    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def main():
    """The main function of the program"""
    args = parse_args()

    # =============================================================================
    #     # serialize model to JSON
    #     model = load_model(os.path.join(args.model_dir, args.model_name), compile=False)
    #     model_json = model.to_json()
    #     with open("model.json", "w") as json_file:
    #         json_file.write(model_json)
    #     # serialize weights to HDF5
    #     model.save_weights("model.h5")
    #     print("Saved model")
    # =============================================================================

    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):
        # load json and create model
        json_file = open(os.path.join(args.model_dir, 'model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(os.path.join(args.model_dir, 'model.h5'))
        log('load trained model on MRI Dataset by Stephen Gregory')
    else:
        model = load_model(os.path.join(args.model_dir, args.model_name), compile=False)
        log('load trained model')

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    for set_cur in args.set_names:

        if not os.path.exists(os.path.join(args.result_dir, set_cur)):
            os.mkdir(os.path.join(args.result_dir, set_cur))
        psnrs = []  # List of Peak Signal to Noise Ratios (PSNRs)
        ssims = []  # List of Structural Similarities (SSIMs)

        for im in os.listdir(os.path.join(args.set_dir, set_cur)):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                # x = np.array(Image.open(os.path.join(args.set_dir,set_cur,im)), dtype='float32') / 255.0
                x = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32) / 255.0
                np.random.seed(seed=0)  # for reproducibility
                y = x + np.random.normal(0, args.sigma / 255.0, x.shape)  # Add Gaussian noise without clipping
                y = y.astype(np.float32)
                y_ = to_tensor(y)
                start_time = time.time()
                x_ = model.predict(y_)  # inference
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))
                x_ = from_tensor(x_)
                psnr_x_ = compare_psnr(x, x_)
                ssim_x_ = compare_ssim(x, x_, multichannel=True)
                # ssim_x_ = compare_ssim(x, x_)
                if args.save_result:
                    name, ext = os.path.splitext(im)
                    show(np.hstack((y, x_)))  # show the image
                    save_result(x_, path=os.path.join(args.result_dir, set_cur,
                                                      name + '_dncnn' + ext))  # save the denoised image
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)

        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)

        if args.save_result:
            save_result(np.hstack((psnrs, ssims)), path=os.path.join(args.result_dir, set_cur, 'results.txt'))

        log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))


if __name__ == '__main__':

    main()
