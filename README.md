# MLOPs Assignment

## Group 41

### Data Versioning with DVC

The ```local``` folder contains the
```.csv.dvc``` file with correct hash code. This folder mimics the developer's system where they can load the dataset. The ```storage``` folder mimics a remote storage machine where different versions of the dataset are managed by dvc.

The dataset can be fetched from ```storage``` to ```local``` folder by setting the ```md5``` attribute of the ```boston_housing.csv.dvc``` file to: 

- **First Version :** ```8ca8328a894f7a9ac5a1565ea1724fda```

- **Second Version :** ```c92319c62910e5be28e53a1ed6522fa7```

Simply run ```dvc pull``` command after setting the ```md5``` attribute.