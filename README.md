# Healthcare-Recommendation-System
## Overview
This project aims to develop a healthcare recommendation system for people in underserved areas with limited access to doctors.  differential diagnosis will be performed using a generative Transformer which will
take a sequence of patient information as input and predict a sequence of most likely pathologies as differential diagnosis, and finally, the most likely pathology will be predicted using a classifier.
![image](https://github.com/gilhotraapoorva/Healthcare-Recommendation-System/assets/129881446/6a989311-8894-45bd-aa28-9063a8ee93bf)

## Usage
1.Clone the repository using the following command:
```
git clone https://github.com/gilhotraapoorva/Healthcare-Recommendation-System.git
```
2. Navigate to the cloned directory:
```
cd Healthcare-Recommendation-System
```
3. Run the inference script:
```
python3 inference.py
```
4. Answer the questions prompted by the system to detect symptoms and provide necessary information.
5. The system will display the detected disease in multiple languages, including English, Hindi, and French.
## Dataset 
The dataset used is of synthetically generated 1.3M patient information that contains patient details (e.g. age, sex), evidence, ground truth differential diagnoses, and condition. The dataset has a total of 49 pathologies that cover various age groups, sexes, and patients with a broad spectrum of medical history.
### About Dataset
We use the term evidence as a general term to refer to a symptom or an antecedent. The dataset contains the following files:

- **release_evidences** a JSON file describing all possible evidences considered in the dataset.
- **release_conditions** a JSON file describing all pathologies considered in the dataset.
- **release_train_patients:** a CSV file containing the patients of the training set.
- **release_validate_patients:** a CSV file containing the patients of the validation set.
- **release_test_patients:** a CSV file containing the patients of the test set.

## About the Files
- ``**dataset.py:**`` generates data loader for training

- ``**questionare.py:**`` Contains the logic for generating questions based on evidences.

- ``**recommendation_system.py:**`` Uses sunthetic dataset to recommend doctor and Hospital based on user ratings.

- ``**network.py:**`` generates proposed network architecture

- ``**train.py:**`` train the network

- ``**test.py:**`` runs the network over the test dataset

- ``**inference.py:**`` runs the inference over a single sample of the dataset

The rest of the files are utility and helper files used to do the preprocessing task.

- ``**preprocess.py:**`` parse the dataset content

- ``**read_utils.py:**`` read condition and evidence information of the dataset

- ``**utils.py:**`` evaluating function utilized during training

- ``**vocab.py:**`` generates vocabulary for both encoder and decoder

## Requirements
- Python 3.x
- PyTorch
- Transformers library
- MarianMTModel
- MarianTokenizer
  
## Contributing
Contributions are welcome! If you encounter any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License. Feel free to use and modify the code as per your requirements.





