import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import pipeline
import  numpy as np

model_name = "xlnet-base-cased"  #her kan vi velge forskjellige modeller fra huggingface
model = XLNetForSequenceClassification.from_pretrained(model_name)
tokenizer = XLNetTokenizer.from_pretrained(model_name)