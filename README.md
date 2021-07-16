# Facial-Mask-Detector
Labels people as 'MASK' or 'NO MASK' in crowded areas in real time, through a model trained using deep learning.<br>
Firstly, we use a KCF TRACKER to track the human faces entering the arena<br>
then we extract faces from each tracking frame and pass them to the trained model<br>
the model then predicts and classifies each person as 'mask' or 'no mask' .
