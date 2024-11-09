import torch
from transformers import Blip2Processor, Blip2Model
from torch.utils.data import DataLoader
from PIL import Image


processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

device = 'cuda'


class VQADataset(torch.utils.data.Dataset):
  def __init__(self, df, data_dir, image_transform=None, text_transform=None):
    """
    Initializes a VQA dataset from a pandas DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing your VQA data. 
            Expected columns: 'image', 'question', 'answer', 'label'.
        image_path (str): Path to the directory containing the images.
        image_transform (callable, optional): Function for image transformation. Defaults to None.
        text_transform (callable, optional): Function for text transformation. Defaults to None.
    """
    self.df = df.copy()  # Avoid modifying the original DataFrame
    self.data_dir = data_dir
    self.image_transform = image_transform
    self.text_transform = text_transform

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    # Get data from the DataFrame
    row = self.df.iloc[idx]
    image_file = row['filename']
    question = row['question']
    answer = row['answer']
    label = row['label']

    # Combine question and answer as text
    text = f"Question: {question} Answer: {answer}"

    # Combine image filename with path
    image_path = self.data_dir + "/" + image_file

    return image_path, text, label


class VQAModel(torch.nn.Module):
  def __init__(self, pretrained_model_name="Salesforce/blip2-opt-2.7b"):
    super(VQAModel, self).__init__()
    self.blip_model = Blip2Model.from_pretrained(pretrained_model_name)
    self.linear_layer = torch.nn.Linear(768, 1)

  def forward(self, inputs):
    outputs = self.blip_model(**inputs)['qformer_outputs'].last_hidden_state[0]
    outputs = outputs[1:, :]  # first token consists edcoding task information [DEC]
    outputs = self.linear_layer(outputs)
    outputs = torch.mean(outputs, dim=0, keepdim=True)  # Reduce to (1, 1)
    
    return outputs
