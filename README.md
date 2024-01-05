# AUTO-MED
AUTO-MED is a web application which can segregate cluttered medical images in a single folder to different categories (each in a seperate folder) and sends the segregated collection of folders as zip file. The wwb application also classifies different types of medical images of different organs and predicts the type of ailment the patient has and can perform image segmentation on specificed medical images. This is an assisting tool for medical practitioners to fasten their progress.

The .h5 files of the models will exist in 'static/models/' folder. Due to size constraints, the .h5 files are not added in the github repository.

The web application was built using Flask, so install the libraries in the ```requirements.txt``` file to run the project using flask.

The web application was used to predict and segregate the following type of organs:
1. Abdomen CT
2. Breast MRI
3. Chest CT
4. CXR
5. Hand
6. Headt CT
7. Kvasir-Capsule
The first 6 data are present in the kaggle dataset "MedMNIST" and Kvasir-Capsule is seperate dataset which can be found online.


