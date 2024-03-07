# DeepArUco++: improved detection of square fiducial markers in challenging lighting conditions
Support code for the DeepArUco++ method. Work by Rafael Berral-Soler, Rafael Muñoz-Salinas, Rafael Medina-Carnicer and Manuel J. Marín-Jiménez.

**NEW (07/03/2024)**: Shadow-ArUco dataset available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10791293.svg)](https://doi.org/10.5281/zenodo.10791293)  
**NEW (06/11/2023)**: Try our method in this Google Colab notebook. Updated to DeepArUco++! [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kR9BYXs9g6N45F-cDiMZ48aR_uxzdEcl?usp=sharing)

To detect markers locally use the demo.py script:

```
python demo.py <path to image> <output path>
```

## Pretrained models
Demo code along with pretrained models will be available soon. 
Some example predictions:

<img src='examples/output_1.png' width='333'> <img src='examples/output_2.png' width='333'> <img src='examples/output_3.png' width='333'>

## Flying-ArUco v2 dataset
This dataset will be available soon. 
Some samples from the dataset:

<img src='examples/flyingaruco_1.jpg' width='333'> <img src='examples/flyingaruco_2.jpg' width='333'> <img src='examples/flyingaruco_3.jpg' width='333'>

## Shadow-ArUco dataset
Dataset available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10791293.svg)](https://doi.org/10.5281/zenodo.10791293)  
Some samples from the dataset:

<img src='examples/shadowaruco_1.png' width='333'> <img src='examples/shadowaruco_2.png' width='333'> <img src='examples/shadowaruco_3.png' width='333'>

## Citation
If you find our work helpful for your research, please cite our publications:
```
@InProceedings{berral2023ibpria,
  author = "Berral-Soler, Rafael
  and Mu{\~{n}}oz-Salinas, Rafael
  and Medina-Carnicer, Rafael
  and Mar{\'i}n-Jim{\'e}nez, Manuel J.",
  editor="Pertusa, Antonio
  and Gallego, Antonio Javier
  and S{\'a}nchez, Joan Andreu
  and Domingues, In{\^e}s",
  title = "DeepArUco: Marker Detection and Classification in Challenging Lighting Conditions",
  booktitle = "Iberian Conference on Pattern Recognition and Image Analysis",
  doi = {10.1007/978-3-031-36616-1_16},
  year = "2023",
  publisher = "Springer Nature Switzerland",
  pages = "199--210",
  isbn = "978-3-031-36616-1"
}
```
