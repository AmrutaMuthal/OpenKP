# Key Phrase extraction
Keyphrase extraction is a language problem represented as: There is a document D in which there are 1-n key phrases which can be used to understand what the document is about, find other relevant documents, and improve many downstream NLP problems.

# Requirements
```
!pip install tokenizers
!pip install transformers
!pip install nltk
!pip install pickle
!pip install tensorflow
!pip install numpy
!pip install matplotlib
import nltk
nltk.download('stopwords')
```

# How to run the demo
Provided all the prerequisite packages are installed.The model can be genrated using the model definition in the models folder. Data processing utils can be accessed from the utils folder. Use the code below to load a checkpoint.Chekpoint link: [ https://drive.google.com/drive/folders/1GmQgNsjGP7OpspjzsMackE89REMDpNLd] The checkpoints at this link are for the large model. For this model l1=120 and l2=12. The demo notebook has the predict_kps function which preprocessed the data and generates KPs. For inspeck checkpoint max_kp=154 and for KP20, max_kp=258. max_len is the max_length of the embedding = 512

```
from models import extractor as ext
from utils import extractionUtils as extUtil

max_len = 512
max_kp = 154

# use model configuration according to training 
tp_model = ext.get_model(max_len,max_kp,l1,l2)

tp_model.load_weights('/path/to/checkpoint/')


```
# For downloading the bert model
Use the code below to download bert uncased. The first line will download the bert model. Locate the vocab.txt file under assets and provide the path in the code below. Link[https://drive.google.com/file/d/1QBdDJc5UIv4sL8WPU3K6pkBEnlGasLoq/]
```
encoder = TFBertModel.from_pretrained("bert-base-uncased")
vocab = "/path/to/vocab.txt"
tokenizer = BertWordPieceTokenizer(vocab, lowercase=True)
```
# Contributing
This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.



