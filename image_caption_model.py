

!pip install tensorflow keras pillow numpy tqdm
! pip install -q kaggle

import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout


from tqdm.notebook import tqdm as tqdm
tqdm().pandas()

# Loading a text file into memory
# def load_doc(filename):
#     # Opening the file as read only
#     file = open(filename, 'r')
#     text = file.read()
#     file.close()
#     return text

import csv
import string


def load_doc(filename):
    file = open(filename, 'r',encoding="utf-8")
    text = file.read()
    file.close()
    return text.lower()





def all_img_captions(filename):
    file = load_doc(filename)

    captions = file.split('\n')
    descriptions = {}
    for captionBulk in captions:
        splitted = captionBulk.split('|')
        if len(splitted) < 3:
            continue
        img = splitted[0].strip()
        comment_number = splitted[1].strip()
        caption = splitted[2]
        if img not in descriptions:
            descriptions[img] = [caption]
        else:
            descriptions[img].append(caption)
    return descriptions






def cleaning_text(captions):
    table = str.maketrans('','',string.punctuation)
    for img,caps in captions.items():
        for i,img_caption in enumerate(caps):

            img_caption.replace("-"," ")
            desc = img_caption.split()
            desc = [word.lower() for word in desc]
            desc = [word.translate(table) for word in desc]
            desc = [word for word in desc if(len(word)>1)]
            desc = [word for word in desc if(word.isalpha())]
            img_caption = ' '.join(desc)
            captions[img][i]= img_caption
    return captions

def text_vocabulary(descriptions):
    vocab = set()

    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]

    return vocab


def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc )
    data = "\n".join(lines)
    file = open(filename,"w")
    file.write(data)
    file.close()

from google.colab import drive
drive.mount('/content/drive')



!unzip /content/drive/MyDrive/dataset30k/dataset30k.zip



dataset_text = "/content/flickr30k_images/results.csv"
dataset_images = "/content/flickr30k_images/flickr30k_images"

filename = dataset_text

descriptions = all_img_captions(filename)
print("Length of descriptions =" , len(descriptions))


clean_descriptions = cleaning_text(descriptions)


vocabulary = text_vocabulary(clean_descriptions)
print("Length of vocabulary = ", len(vocabulary))



save_descriptions(clean_descriptions, "CAPYOPN.txt")

vocabulary

descriptions

clean_descriptions

descriptions

import os
def extract_features(directory):
    model = Xception(include_top=False, pooling='avg')
    features = {}
    for img in tqdm(os.listdir(directory)):
        filename = os.path.join(directory, img)


        if os.path.splitext(filename)[1].lower() in ('.jpg', '.jpeg', '.png'):
            image = Image.open(filename)
            image = image.resize((299, 299))
            image = np.expand_dims(image, axis=0)
            image = image / 127.5
            image = image - 1.0

            feature = model.predict(image)
            features[img] = feature
    return features


features = extract_features(dataset_images)
dump(features, open("/content/drive/MyDrive/features.p","wb"))

features = load(open("/content/drive/MyDrive/features.p","rb"))


def load_photos(filename):
  file = load_doc(filename)
  photos=list()
  for line in file.split('\n'):
    if len(line) < 1:
      continue
    identifier = line.split('|')[0]
    photos.append(identifier)
  return set(photos)



def load_clean_descriptions(filename, train_imgs_photos):
    try:
        file = load_doc(filename)
        descriptions = {}

        for line in file.split("\n"):
            words = line.split("\t")
            if len(words) < 1:
                continue

            image_fid, image_caption = words[0], words[1:]

            #print (f"image {image_fid} | caption {image_caption}")


            if os.path.splitext(image_fid)[1].lower() in ('.jpg', '.jpeg', '.png'):
              if image_fid in train_imgs_photos:
                  if image_fid not in descriptions:
                      descriptions[image_fid] = []
                  desc = '<start> ' + " ".join(image_caption) + ' <end>'
                  descriptions[image_fid].append(desc)
              else:
                  print(f"Warning: Image '{train_imgs_photos[1]}' not found in photos.")
        return descriptions
    except Exception as e:
        print(f"Error loading descriptions: {e}")
        return {}


def load_features(photos):
    all_features = load(open("/content/drive/MyDrive/features.p", "rb"))
    features = {}
    for k in photos:
        if k in all_features:
            features[k] = all_features[k]
        else:
            print(f"Warning: Key '{k}' not found in all_features.")
    return features

filename = dataset_text


train_imgs = load_photos(filename)
train_descriptions = load_clean_descriptions("/content/CAPYOPN.txt", train_imgs)
train_features = load_features(train_imgs)


def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


from keras.preprocessing.text import Tokenizer

def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer


tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('/content/drive/MyDrive/tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
vocab_size


def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)

max_length = max_length(descriptions)
max_length


def data_generator(descriptions, features, tokenizer, max_length):
    while 1:
        for key, description_list in descriptions.items():
            
            feature = features[key][0]
            input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, description_list, feature)
            yield ([input_image, input_sequence], output_word)

def create_sequences(tokenizer, max_length, desc_list, feature):
    X1, X2, y = list(), list(), list()
    
    for desc in desc_list:
        
        seq = tokenizer.texts_to_sequences([desc])[0]
        
        for i in range(1, len(seq)):
            
            in_seq, out_seq = seq[:i], seq[i]
            
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)


[a,b],c = next(data_generator(train_descriptions, features, tokenizer, max_length))
a.shape, b.shape, c.shape

train_descriptions

from tensorflow.keras.utils import plot_model


def define_model(vocab_size, max_length):

    
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    
    print(model.summary())
    plot_model(model, to_file='/content/drive/MyDrive/model.png', show_shapes=True)

    return model


print('Dataset: ', len(train_imgs))
print('Descriptions: train=', len(train_descriptions))
print('Photos: train=', len(train_features))
print('Vocabulary Size:', vocab_size)
print('Description Length: ', max_length)

model = define_model(vocab_size, max_length)
print(model,'model')
epochs = 10
steps = len(train_descriptions)

os.mkdir("/content/drive/MyDrive/models")
for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save("/content/drive/MyDrive/models/model_" + str(i) + ".h5")