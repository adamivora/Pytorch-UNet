import requests
from predict import predict
from pathlib import Path


def download_model():
    model_path = Path('MODEL.pth')
    if model_path.exists():
        return
    print('Downloading model...')
    url = 'https://github.com/milesial/Pytorch-UNet/releases/download/v1.0/unet_carvana_scale1_epoch5.pth'
    r = requests.get(url, allow_redirects=True)

    open(model_path, 'wb').write(r.content)
    print(f'Saved model to {model_path}.')


if __name__ == '__main__':
    download_model()

    inputs_path = Path('input')
    outputs_path = Path('output')
    outputs_path.mkdir(exist_ok=True, parents=True) # create directory if not exists

    frames = list(inputs_path.glob('*.jpg')) # get all *.jpg files from the input directory
    outputs = [outputs_path.joinpath(img.name) for img in frames]

    print('Input images:', frames)
    print('Output images:', outputs)

    predict(frames, outputs)
    print('Done.')
