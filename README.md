
# Feature extraction for palpebral conjunctiva - 01

This library is an implementation of the feature extraction described in the paper:  
Y.-M. Chen, S.-G. Miaou, and H. Bian, “Examining palpebral conjunctiva for anemia assessment with image processing methods,”   
Comput. Methods Programs Biomed., vol. 137, pp. 125–135, Dec. 2016.  
  
Coded by franklin paul barrientos porras (frank2207a@gmail.com)   
and  Bryan Percy Saldivar Espinoza (bsaldivar.emc2@gmail.com)  
  
Status of development: Under construction  
  
It requires:  
    opencv 3.2.0  
    numpy 1.11.3  
    matplotlib 2.0.0  



```
import imp
import miau_features
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import conj12
from conj12 import ConjFeat
imp.reload(miau_features)
imp.reload(conj12)
```




    <module 'conj12' from '/home/bsaldivar/MEGA/Documents/Jobs/UPCH/Anemia/20180208/conj12.py'>



Uploading an image


```
img = 'Images/IMG_XYZ.jpg'
imp = Image.open(img)
imn_ = np.asarray(imp)
print(imn_.shape)

```

    (4160, 3120, 3)
    (20, 20, 3)


Create the object for feature extraction


```
#Create object specifying the image and channel order, default bgr, otherwise specify.
cf = ConjFeat(imn_,input_channels='rgb')

#Extract the features. Set v=True for printing the generated values
values = cf.get_features(v=True)

#The features are returned as a dictionary
print(values.keys())
```

    Entropy: 1.61164710202
    HHR: 18.584999999999972
    PVM [[ 0.50617284]
     [ 4.49382716]
     [ 4.        ]]
    Brightness 2.1494471175
    PVM-12 percentile [[ 0.96        0.          1.          2.87804878]
     [ 4.5725      2.925       4.97560976  6.63414634]
     [ 3.9325      2.35        4.          5.80487805]]





    dict_keys(['HHR', 'Brightness', 'Entropy', 'PVM', 'PMV_12'])




```
"""
#Use these for a manual calculation of every feature
cf.calc_entropy(v=True)
cf.calc_HHR(v=True)
cf.calc_PVM(v=True)
cf.calc_bright(v=True)
cf.calc_PVMper(v=True)
"""
```
