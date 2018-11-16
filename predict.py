import argparse
import torch
from torchvision import datasets, transforms, models
from PIL import Image
from train import setup_network

# TODO:
# - Not use top_k if there is no top_k
# - use cahtegory_names

def predict(image_path, checkpoint, category_names, top_k):
  top_k = top_k if top_k is not None else 5
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  image = image_to_numpy(image_path)
  model = load_model_from_checkpoint(checkpoint)
  model.to(device)
  with torch.no_grad():
      # https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
      image = image.to(device).unsqueeze_(0)

      output = model(image).topk(top_k)
      probabilities = torch.nn.functional.softmax(output[0].data, dim=1).cpu().numpy()[0]
      classes = output[1].data.cpu().numpy()[0]
      print(probabilities)
      print(classes)
      # return probabilities.topk(top_k)

def load_model_from_checkpoint(checkpoint):
  checkpoint = torch.load('{}'.format(checkpoint))
  label_count = len(checkpoint['class_to_idx'])
  model_name = checkpoint['arch_name']
  hidden_units = checkpoint['hidden_units']
  model, _, _ = setup_network(model_name, hidden_units, label_count)
  model.class_to_idx = checkpoint['class_to_idx']
  model.load_state_dict(checkpoint['state_dict'])

  return model

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
