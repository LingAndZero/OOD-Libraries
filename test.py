import gdown
import os


download_id_dict = {
    'osr': '1L9MpK9QZq-o-JrFHrfo5lM4-FsFPk0e9',
    'mnist_lenet': '13mEvYF9rVIuch8u0RVDLf_JMOk3PAYCj',
    'cifar10_res18': '1rPEScK7TFjBn_W_frO-8RSPwIG6_x0fJ',
    'cifar100_res18': '1OOf88A48yXFw4fSU02XQT-3OQKf31Csy',
    'imagenet_res50': '1tgY_PsfkazLDyI1pniDMDEehntBhFyF3',
    'cifar10_res18_v1.5': '1byGeYxM_PlLjT72wZsMQvP6popJeWBgt',
    'cifar100_res18_v1.5': '1s-1oNrRtmA0pGefxXJOUVRYpaoAML0C-',
    'imagenet200_res18_v1.5': '1ddVmwc8zmzSjdLUO84EuV4Gz1c7vhIAs',
    'imagenet_res50_v1.5': '15PdDMNRfnJ7f2oxW6lI-Ge4QJJH3Z0Fy',
    'benchmark_imglist': '1lI1j0_fDDvjIt9JlWAw09X8ks-yrR_H1',
    'usps': '1KhbWhlFlpFjEIb4wpvW0s9jmXXsHonVl',
    'cifar100': '1PGKheHUsf29leJPPGuXqzLBMwl8qMF8_',
    'cifar10': '1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1',
    'cifar10c': '170DU_ficWWmbh6O2wqELxK9jxRiGhlJH',
    'cinic10': '190gdcfbvSGbrRK6ZVlJgg5BqqED6H_nn',
    'svhn': '1DQfc11HOtB1nEwqS4pWUFp8vtQ3DczvI',
    'fashionmnist': '1nVObxjUBmVpZ6M0PPlcspsMMYHidUMfa',
    'cifar100c': '1MnETiQh9RTxJin2EHeSoIAJA28FRonHx',
    'mnist': '1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb',
    'fractals_and_fvis': '1EZP8RGOP-XbMsKex3r-BGI5F1WAP_PJ3',
    'tin': '1PZ-ixyx52U989IKsMA2OT-24fToTrelC',
    'tin597': '1R0d8zBcUxWNXz6CPXanobniiIfQbpKzn',
    'texture': '1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam',
    'imagenet10': '1qRKp-HCLkmfiWwR-PXthN7-2dxIQVKxP',
    'notmnist': '16ueghlyzunbksnc_ccPgEAloRW9pKO-K',
    'places365': '1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay',
    'places': '1fZ8TbPC4JGqUCm-VtvrmkYxqRNp2PoB3',
    'sun': '1ISK0STxWzWmg-_uUr4RQ8GSLFW7TZiKp',
    'species_sub': '1-JCxDx__iFMExkYRMylnGJYTPvyuX6aq',
    'imagenet_1k': '1i1ipLDFARR-JZ9argXd2-0a6DXwVhXEj',
    'ssb_hard': '1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE',
    'ninco': '1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6',
    'imagenet_v2': '1akg2IiE22HcbvTBpwXQoD7tgfPCdkoho',
    'imagenet_r': '1EzjMN2gq-bVV7lg-MEAdeuBuz-7jbGYU',
    'imagenet_c': '1JeXL9YH4BO8gCJ631c5BHbaSsl-lekHt',
    'imagenet_o': '1S9cFV7fGvJCcka220-pIO9JPZL1p1V8w',
    'openimage_o': '1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE',
    'inaturalist': '1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj',
    'actmed': '1tibxL_wt6b3BjliPaQ2qjH54Wo4ZXWYb',
    'ct': '1k5OYN4inaGgivJBJ5L8pHlopQSVnhQ36',
    'hannover': '1NmqBDlcA1dZQKOvgcILG0U1Tm6RP0s2N',
    'xraybone': '1ZzO3y1-V_IeksJXEvEfBYKRoQLLvPYe9',
    'bimcv': '1nAA45V6e0s5FAq2BJsj9QH5omoihb7MZ',
}


def download_dataset(dataset):

    store_path = './data/' + dataset + '.zip'
    gdown.download(id=download_id_dict[dataset], output=store_path, proxy="http://10.82.60.184:40903")


download_dataset('openimage_o')