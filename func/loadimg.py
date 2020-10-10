import os
from PIL import Image
import json
import numpy as np
from func.utils import preprocess_image

# Used to get a list of pic_path and its corresponding CNN prediction index

def get_file_path():
    return os.path.split(os.path.realpath(__file__))[0]

filepath = os.path.abspath(os.path.join(get_file_path(), ".."))+"/data"

# ----------------------------------------
def find_which_label_by_n(n, label_index_dict):
    for i in range(len(label_index_dict)):
        if n==label_index_dict[str(i)][0]:
            return i

def get_label_index_dict(filepath = filepath+"/imagenet_label_index.json"):
    with open(filepath, 'r') as f:
        label_index_dict = json.load(f)
    
    return label_index_dict

def generate_example_list(label_index_dict_filepath = filepath+"/imagenet_label_index.json",
                          pic_filepath = filepath+"/images"):
    label_index_dict = get_label_index_dict(label_index_dict_filepath)
    
    pic_names=os.listdir(pic_filepath)
    pic_names.sort()
    
    path_label_index_record=[]

    for idx_of_pic,pic in enumerate(pic_names):
        index = find_which_label_by_n(pic.split('_')[0], label_index_dict)
        
        if index!=None:
            path_label_index_record.append([pic_filepath+'/'+pic, index])
    
    return path_label_index_record

def get_example_pic_and_preprocess(example_index=None, label_index_dict_filepath = filepath+"/imagenet_label_index.json", pic_filepath = filepath+"/images"):
    
    example_list = generate_example_list(label_index_dict_filepath, pic_filepath)
    
    if example_index==None:
        example_index=np.random.randint(0, len(example_list))
    
    assert example_index>=0 and example_index<=len(example_list)
    
    print(str(example_index)+" "+str(example_list[example_index])+' is chosen.')
    
    img_path = example_list[example_index][0]
    target_class_index = example_list[example_index][1]
    
    file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
    
    # Read image
    original_image = Image.open(img_path).convert('RGB')
    
    # Process image
    processed_img_tensor, img_resize = preprocess_image(original_image)
    
    return (original_image,
            img_resize,
            processed_img_tensor,
            target_class_index,
            file_name_to_export)
# ----------------------------------------