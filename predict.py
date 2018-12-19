import argparse
import torch
from torchvision import datasets, transforms, models
from PIL import Image
from train import load_model
import torch.nn.functional as F
import json

# TODO:
# - Not use top_k if there is no top_k
# - use category_names

def predict(image_path, checkpoint, category_names, top_k):
    top_k = top_k if top_k is not None else 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image_to_numpy(image_path)
    checkpoint_dict = torch.load(checkpoint)
    # GOTCHA: Not sure why I used structure, must have been somewhere
    # I read, but should be arch_name
    arch_name = checkpoint_dict['structure']
    num_labels = len(checkpoint_dict['class_to_idx'])
    # GOTCHA: both learning rate and hidden_units should be in the
    # dict, but they weren't and there's not time to retrain the whole model 
    hidden_units = 4096
    learning_rate = 0.001
    model = load_model(arch_name, learning_rate, hidden_units, num_labels)
    model.to(device)
    with torch.no_grad():
        # https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
        image = image.to(device).unsqueeze_(0)
        image = image.float()
        top_k_results = model(image).topk(top_k)
        
        # convert torch => numpy
        # https://github.com/pytorch/vision/issues/432#issuecomment-368330817
        if str(device) == 'cuda':
            probs = torch.nn.functional.softmax(top_k_results[0].data, dim=1).cpu().numpy()[0]
            classes = top_k_results[1].data.cpu().numpy()[0]
        else:
            probs = torch.nn.functional.softmax(top_k_results[0].data, dim=1).numpy()[0]
            classes = top_k_results[1].data.numpy()[0]
        
        with open('cat_to_name.json', 'r') as f:
            cats = json.load(f)

        labels = list(cats.values())
        class_labels = [labels[x] for x in classes]
        print('-----------------------')
        print('Prediction, probability')
        print('-----------------------')
        print (list(zip(class_labels, probs)))

def image_to_numpy(image_path):
  image_norm_mean = [0.485, 0.456, 0.406]
  image_norm_std = [0.229, 0.224, 0.225]

  image = Image.open(image_path)
  img_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(image_norm_mean,
                                                            image_norm_std)])

  return img_transforms(image)



# CMD
parser = argparse.ArgumentParser()
parser.add_argument('image',
                    type=str,
                    help='Show us your image?')
parser.add_argument('checkpoint',
                    type=str,
                    help='Checkpoint please?')
parser.add_argument('--category_names',
                    type=int,
                    help='List of real labels to map predictions to',
                    default=None)
parser.add_argument('--top_k',
                    type=int,
                    help='Amount of nearest predictions to show?')

args, _ = parser.parse_known_args()
predict(
  args.image,
  args.checkpoint,
  args.category_names,
  args.top_k)
