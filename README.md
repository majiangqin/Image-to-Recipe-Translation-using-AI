# Deep Image-to-Recipe Translation
*Team DeepChef (Bilal Mawji, Franz D Williams, Jiangqin Ma)*  
*CS 7643 Final Project*  

## Summary of files
* `deepchef.ipynb` - Iterations on our custom model and recipe step prediction process
* `data_augmentation.ipynb` - Image data augmentation pipeline development
* `graphs.ipynb` - Generates graphs based on collected run data
* `train.ipynb` - Trains ResNet50-powered ingredient predictor models
* `train_custom_model.ipynb` - Trains custom CNN ingredient predictor models
* `project/` - Contains data loading, image augmentation, metric, and model building utilities used by several of the project notebooks above

## Setup
Install requirements:

```
pip install -r requirements.txt
```

Download the dataset from Kaggle: [Food Ingredients and Recipes Dataset with Images](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images), and extract the contents (`Food Images` directory and the CSV file) to `project/dataset/`

---

### A note on Git history
We initially found [a blog post related to our idea](https://towardsdatascience.com/this-ai-is-hungry-b2a8655528be) and loaded its associated code into our repo. However, we ended up instead taking inspiration from Facebook/Meta's [Inverse Cooking paper](https://research.facebook.com/publications/inverse-cooking-recipe-generation-from-food-images/), and did not use any of the initial blog post code we imported.
