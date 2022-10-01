import torch
from model.network import SCOAD
from misc import config
from core.interface import test_full
from Dataset import ThumosDataset

torch.set_default_tensor_type('torch.cuda.FloatTensor')

if __name__ == '__main__':

    args = config.parse_args()
    device = torch.device('cuda:' + str(args.cuda_id))
    dataset = ThumosDataset(args)

    model = SCOAD(args).to(device)
    model.load_state_dict(torch.load(args.pretrained_ckpt), strict = False)

    model.eval()
    test_full(dataset, model, device, args.fps)