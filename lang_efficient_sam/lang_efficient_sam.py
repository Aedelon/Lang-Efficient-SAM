import groundingdino.datasets.transforms as T
import torch
import numpy as np
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from torchvision.transforms import ToTensor
from huggingface_hub import hf_hub_download
import time

def load_model_hugging_face(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(f"Model loaded from {cache_file} \n => {log}")
    model.eval()
    return model


class LangEfficientSAM:
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        print("Device:", self.device)
        if self.device == "cpu":
            self.sam_efficient = torch.jit.load('./models/efficientsam_s_cpu.jit')
        else:
            self.sam_efficient = torch.jit.load('./models/efficientsam_s_gpu.jit')
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino = load_model_hugging_face(ckpt_repo_id,
                                                     ckpt_filename,
                                                     ckpt_config_filename,
                                                     self.device)

    def predict_dino(self, image_pil, text_prompt, box_threshold, text_threshold):
        start = time.time()
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = transform(image_pil, None)

        boxes, logits, phrases = predict(model=self.groundingdino,
                                         image=image_transformed,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         device=self.device)
        W, H = image_pil.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        # print("DINO time: ", time.time() - start)

        return boxes, logits, phrases

    def predict_sam(self, image, box):
        start = time.time()
        img_tensor = ToTensor()(image).to(device=self.device)
        bbox = torch.reshape(box.clone().detach(), [1, 1, 2, 2]).to(device=self.device)
        bbox_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2]).to(device=self.device)

        predicted_logits, predicted_iou = self.sam_efficient(
            img_tensor[None, ...],
            bbox,
            bbox_labels,
        )
        predicted_logits = predicted_logits.cpu()
        all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
        predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()

        max_predicted_iou = -1
        selected_mask_using_predicted_iou = None
        for m in range(all_masks.shape[0]):
            curr_predicted_iou = predicted_iou[m]
            if (
                    curr_predicted_iou > max_predicted_iou
                    or selected_mask_using_predicted_iou is None
            ):
                max_predicted_iou = curr_predicted_iou
                selected_mask_using_predicted_iou = all_masks[m]

        # print("SAM time: ", time.time() - start)
        return selected_mask_using_predicted_iou

    def predict(self, image_pil, text_prompt, box_threshold=0.3, text_threshold=0.25):
        boxes, logits, phrases = self.predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
        # masks = torch.tensor([])
        masks = []
        if len(boxes) > 0:
            for box in boxes:
                mask = self.predict_sam(image_pil, box)
                masks.append(mask)

        masks = np.array(masks)
        masks = torch.from_numpy(masks)

        return masks, boxes, phrases, logits
